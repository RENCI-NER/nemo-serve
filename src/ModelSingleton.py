import asyncio
import logging
import math
import os
import re
import sys
import yaml
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from starlette.concurrency import run_in_threadpool
# SAPRedis/SAPQdrant are imported lazily in SapbertModelWrapper.__init__ rather
# than here. They pull in redis and qdrant_client, which are absent from the
# token-classification images (ghcr.io/renci-ner/nemo-serve:v1.3.1 and friends);
# importing them at module scope makes this file unimportable there and takes the
# whole app down, sapbert backend or not. Same reason `nemo` is imported inside
# TokenClassificationModelWrapper.__init__.
from src.utils.tokenizer import tokenizer

import yaml


logger = logging.Logger("gunicorn.error")


class ModelNotFoundError(Exception):
    pass


def _cgroup_cpu_limit():
    """The container's CPU limit, or None when unlimited / not in a cgroup.

    os.cpu_count() reports the host's cores, which on a 96-core node bears no
    relation to a 4-CPU pod.
    """
    try:
        # cgroup v2, then v1. ponytail: files only, no dependency on a lib.
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        return None if quota == "max" else int(quota) / int(period)
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        return None if quota <= 0 else quota / period
    except OSError:
        return None


def _cap_torch_threads():
    """torch sizes its thread pool from os.cpu_count(), i.e. the host's cores,
    not the container's CPU limit. On a 96-core node with a 4-CPU limit that is
    48 OpenMP threads fighting over 4 CPUs of quota: measured 880ms per SapBERT
    forward pass vs 56ms at 4 threads.

    OMP_NUM_THREADS in the pod spec is the primary fix; this is the backstop for
    when it is missing.
    """
    if os.environ.get("OMP_NUM_THREADS"):
        return  # operator has spoken
    limit = _cgroup_cpu_limit()
    if limit:
        torch.set_num_threads(max(1, int(limit)))
        logger.info("Capped torch threads to %d from cgroup CPU limit %.2f",
                    torch.get_num_threads(), limit)


class ModelWrapper:
    """ Inherit this class and do any model intialization here"""
    def __init__(self):
        pass

    def __call__(self, query_text):
        """ Make a call to initialized model's Predict function"""
        raise NotImplementedError("Call to wrapped model is required")


class TokenClassificationModelWrapper(ModelWrapper):
    def __init__(self, model_path):
        """ Initializes NLP Model
        :param model_path: Path to model to load
        """
        from nemo.collections.nlp.models import TokenClassificationModel
        from omegaconf import open_dict
        super(TokenClassificationModelWrapper, self).__init__()
        self.model = TokenClassificationModel.restore_from(model_path)
        # NeMo's _infer builds a fresh DataLoader per call and reads
        # num_workers off the checkpoint's config -- medmentions-v0.2.nemo ships
        # num_workers=2. Forking two worker processes to feed a handful of
        # sentences costs a measured 130ms per _infer call vs 0.8ms at 0
        # workers. Inference batches are tiny; there is nothing to prefetch.
        with open_dict(self.model._cfg):
            self.model._cfg.dataset.num_workers = 0
        # _infer captures self.training and restores it in a finally block. Pin
        # eval mode once here so that restore is a no-op and concurrent calls
        # from the threadpool can't flip the model into train mode mid-forward.
        self.model.eval()
        # GPU work serialises on the CUDA stream anyway; 2 slots let one
        # request's tokenisation/post-processing overlap another's forward pass
        # without oversubscribing.
        self._infer_slots = asyncio.Semaphore(2)
        # Make this an instance variable so that it's easier to mock in
        # testing:
        self.sentence_tokenizer = tokenizer

        # The model can only accept sequences up to max_seq_length tokens,
        # INCLUDING the [CLS]/[SEP] special tokens the model adds internally.
        # `text_to_tokens` (used throughout chunking) does NOT count those, so
        # we reserve 2 slots for them. Feeding a chunk longer than the model's
        # limit produces out-of-range indices in a CUDA kernel, which triggers
        # a device-side assert that permanently corrupts the CUDA context for
        # the life of this process, so every chunk MUST stay under this.
        self.max_seq_length = self.model._cfg.dataset.max_seq_length
        self.window_size = self.max_seq_length - 2
        # _pubannotate reads these once per WORD. Every read walks omegaconf's
        # DictConfig resolution machinery (~29us a hit, measured 2.3s of the
        # 2.5s spent annotating a 200-sentence document). They never change
        # after load, so resolve them here and read plain Python objects in the
        # loop.
        self.ids_to_labels = {v: k
                              for k, v in self.model._cfg.label_ids.items()}
        self.pad_label = self.model._cfg.dataset.pad_label
        logger.info("Model max_seq_length=%d, using content window_size=%d",
                    self.max_seq_length, self.window_size)

    def _get_token_length(self, input_text):
        """Return the length in tokens as understood by the model's own internal
        tokenizer.
        """
        tokens = self.model.tokenizer.text_to_tokens(input_text)
        return len(tokens)

    def _token_chunks(self, input_text, window_size=512):
        """Break sentences into chunks. Yields a tuple of token count, string.

        Each chunk will be smaller than window_size tokens.
        This function is a generator
        """
        logger.debug("_token_chunks called on text: %s", input_text)
        tokens = self.model.tokenizer.text_to_tokens(input_text)
        token_count = len(tokens)
        logger.debug("Text tokenized to %d tokens", token_count)

        if token_count < window_size:
            # Chunk has been sufficiently split before now. Just kick it back.
            yield (token_count, input_text)

        else:
            # Walk the words greedily, packing as many as fit under window_size
            # and flushing a group just before it would overflow.
            words = input_text.split()
            if not words:
                logger.debug("Zero tokens found, returning None")
                return

            word_token_counts = self._word_token_counts(tokens, words)

            group = []
            group_tokens = 0
            for word, word_tokens in zip(words, word_token_counts):
                if group and group_tokens + word_tokens >= window_size:
                    # Adding this word would overflow; flush the current group.
                    yield (group_tokens, " ".join(group) + " ")
                    group = []
                    group_tokens = 0
                group.append(word)
                group_tokens += word_tokens
            if group:
                # A lone word longer than the window can't be split further; it
                # is truncated in _predict_all before it reaches the model.
                yield (group_tokens, " ".join(group) + " ")

    @staticmethod
    def _word_token_counts(tokens, words):
        """Map the flat list of wordpiece tokens back onto how many tokens each
        whitespace word contributed, in a single pass (no extra tokenizer
        calls). Wordpiece continuation tokens start with '##'; each token that
        is NOT a continuation marks the start of a new word.

        Returns a list of per-word token counts aligned with `words`. If the
        tokenizer's word boundaries can't be reconciled with whitespace words
        (unexpected), falls back to an even split so callers still get a usable
        estimate rather than an error.
        """
        counts = []
        for tok in tokens:
            if tok.startswith("##") and counts:
                counts[-1] += 1
            else:
                counts.append(1)
        if len(counts) != len(words):
            # Boundaries didn't line up (e.g. punctuation split differently).
            # Distribute total tokens roughly evenly; correctness is still
            # guaranteed by the final _truncate_to_window safety net.
            total = len(tokens)
            per = max(1, total // max(1, len(words)))
            counts = [per] * len(words)
        return counts

    def _sentences_to_chunks(self, sentences, window_size):
        """
        Take a list of sentences, return an array of lists of otkens that are
        all smaller than window_size.
        """
        for sentence in sentences:
            sentence_token_length = self._get_token_length(sentence)
            if sentence_token_length >= window_size:
                logger.debug("Found an extra-long sentence, "
                             "breaking it up:\n%s\n", sentence)

                # Try splitting on semicolons into sentence fragments
                # re.split with a lookbehind pattern includes the semicolon on
                # the split text.
                split_list = re.split(r'(?<=\;\s)|(?<=\;)', sentence)
                logger.debug("Semicolon split broke it into %d pieces",
                             len(split_list))

                for fragment in split_list:
                    # Break any fragment over window_size into bits.
                    yield from self._token_chunks(fragment, window_size)
            else:
                logger.debug("Sentence is under chunk size (has %d tokens, "
                             "returning: %s", sentence_token_length, sentence)
                yield (sentence_token_length, sentence)

    def sliding_window(self, input_text, window_size=512):
        """
        Tokenize original query into smaller chunks that the model can process

        This refactored function uses a stack data structure instead of a
        rolling window.
        :param text: Text to split up
        :param window_size: Max token size to split
        :return: Array of split text
        """
        logger.debug("sliding window called with text of length %d chars",
                     len(input_text))
        sentences = self.sentence_tokenizer.tokenize(input_text)
        logger.debug("Text broken into %d sentences", len(sentences))

        current_string = ""
        current_token_length = 0
        for (chunk_token_length, chunk) in self._sentences_to_chunks(
                sentences, window_size):
            logger.debug("sliding_window is working on sentence chunk %s",
                         str(chunk))
            if current_token_length + chunk_token_length >= window_size:
                # New sentence would make the chunk too long. Yield the
                # existing chunk and start a new chunk.
                if current_string:
                    yield current_string
                current_string = chunk
                current_token_length = chunk_token_length
            else:
                current_string += chunk
                current_token_length += chunk_token_length
                logger.debug("current_string is %d tokens long",
                             current_token_length)
        yield current_string

    def _pubannotate(self, q, inferred):
        queries = [q.strip().split() for q in q]
        ids_to_labels = self.ids_to_labels
        pad_label = self.pad_label
        start_idx = 0
        end_idx = 0
        denotations = []
        for query in queries:
            end_idx += len(query)
            # extract predictions for the current query from the list of all
            # predictions
            preds = inferred[start_idx:end_idx]
            start_idx = end_idx
            for j, word in enumerate(query):
                # strip out the punctuation to attach the entity tag to the word
                # not to a punctuation mark that follows the word
                if not word[-1].isalpha():
                    word = word[:-1]
                pad = 0 if j == 0 else 1
                span_start = len(' '.join(query[:j])) + pad
                span_end = span_start + len(word)

                label = ids_to_labels[preds[j]]

                is_not_pad_label = (label != pad_label and label != '0')

                if not is_not_pad_label:
                    # For things like fitness to practice where model labels it as fitness[B-biolink:NamedThing] to[0] # practice[I-biolink:NamedThing]
                    # @TODO: investigate why [ De no ##vo ma ##li ##gna ##ncy following re ##nal transplant ##ation ] would return I-biolink without a B-
                    if len(denotations) and j + 1 < len(query) and ids_to_labels[preds[j + 1]].startswith('I-'):
                        denotations[-1]['text'] += " " + word
                else:
                    if label.startswith('I-') and len(denotations):
                        denotations[-1]['span']['end'] = span_end
                        denotations[-1]['text'] += " " + word
                    else:
                        label = label.replace('B-', '').replace('I-', '')
                        denotation = {
                            'id': f'I{j}-',
                            'span': {
                                'begin': span_start,
                                'end': span_end
                            },
                            'obj': label,
                            'text': word
                        }
                        denotations.append(denotation)
        return {
            'text': ''.join(q),
            'denotations': denotations
        }

    def _predict_all(self, queries, batch_size: int = 32):
        """Annotate every chunk of one document in a single _infer call.

        _infer has a fixed per-call cost (DataLoader construction, eval/train
        toggling, a cuda sync) and batches internally, so calling it once per
        chunk paid that cost N times and left the GPU running batches of one.
        _infer returns one prediction per whitespace word, concatenated in
        query order, so the flat result slices cleanly back per chunk --
        _pubannotate's own span offsets are chunk-relative, and
        _merge_pub_annotator_annotations re-bases them onto the full text.
        """
        # _truncate_to_window is a last-resort safety net. Chunking already
        # packs every query under the model limit word-by-word, so it normally
        # does nothing (fast path returns the query unchanged). It only bites
        # when a SINGLE whitespace word tokenizes to >= window_size tokens --
        # something that can't be split on word boundaries and essentially
        # never occurs in real text. In that lone case we drop the word's tail
        # rather than let an over-length sequence reach the model and trigger a
        # CUDA device-side assert that would take down the whole server. Losing
        # annotations on one pathological word is strictly better than crashing
        # for everyone.
        safe_queries = [self._truncate_to_window(q) for q in queries
                        if q and q.strip()]
        if not safe_queries:
            return {"text": "", "denotations": []}
        inferred = self.model._infer(safe_queries, batch_size)
        per_chunk = []
        start = 0
        for query in safe_queries:
            end = start + len(query.strip().split())
            per_chunk.append(self._pubannotate([query], inferred[start:end]))
            start = end
        return self._merge_pub_annotator_annotations(per_chunk)

    def _truncate_to_window(self, query):
        """
        Return query trimmed to at most window_size model tokens (whole words
        are dropped from the end, since token labels map back to whole words).
        """
        tokens = self.model.tokenizer.text_to_tokens(query)
        if len(tokens) <= self.window_size:
            return query
        words = query.split()
        word_token_counts = self._word_token_counts(tokens, words)
        kept = []
        running = 0
        for word, count in zip(words, word_token_counts):
            if running + count > self.window_size:
                break
            kept.append(word)
            running += count
        logger.warning(
            "Query over window (%d tokens > %d); truncated to %d words",
            len(tokens), self.window_size, len(kept))
        return " ".join(kept)

    @staticmethod
    def _merge_pub_annotator_annotations(annotations):
        result = {
            "text": "",
            "denotations": []
        }
        for index, a in enumerate(annotations):
            if index == 0:
                result = a
                continue
            offset = len(result['text'])
            denotations = a['denotations']
            new_denotations = [{
                'id': span['id'] + f'{index}',
                'span': {
                    'begin': span['span']['begin'] + offset,
                    'end': span['span']['end'] + offset
                },
                'obj': span['obj'],
                'text': span['text']
            } for span in denotations]
            result['text'] += a['text']
            result['denotations'] += new_denotations
        return result

    async def __call__(self, query_text, *args, **kwargs):
        """ Runs prediction on text"""
        try:
            queries = [x for x in self.sliding_window(query_text,
                                                      self.window_size)]
            # Inference is sync and CPU/GPU-bound: running it inline in an
            # async def blocks the event loop, which serialises every other
            # request (and starves the health checks) for its whole duration.
            async with self._infer_slots:
                return await run_in_threadpool(self._predict_all, queries)
        except RuntimeError as E:
            # A CUDA device-side assert (e.g. an out-of-range index from an
            # over-length input) permanently corrupts the CUDA context for the
            # whole process: every subsequent request would get the same error.
            # Toggling train mode does NOT recover it. The only real fix is a
            # fresh process, so exit hard and let the orchestrator (k8s) restart
            # this pod with a clean context instead of serving errors forever.
            if "CUDA" in str(E) or "device-side assert" in str(E):
                logger.error("Unrecoverable CUDA error, exiting to force "
                             "restart: %s", E)
                sys.stderr.flush()
                sys.stdout.flush()
                os._exit(1)
            raise
        finally:
            # reset the model, recover
            self.model.train(mode=self.model.training)


class TokenClassificationModelWrapperMock(ModelWrapper):
    def __init__(self, model_path):
        """ Initializes NLP Model"""
        print('hey')

    def __call__(self, query_text):
        """ Runs prediction on text"""
        return ['woop']


class SapbertModelWrapper(ModelWrapper):

    def __init__(self, model_path, connection_config, backend='redis'):
        """ Initializes NLP Model"""
        super(SapbertModelWrapper, self).__init__()
        _cap_torch_threads()
        # One forward pass already occupies torch.get_num_threads() CPUs, so
        # only cpus/threads of them fit at once. starlette's threadpool defaults
        # to 40 workers, which on a 4-CPU pod means up to 160 OS threads and the
        # same oversubscription _cap_torch_threads exists to prevent. Measured
        # on a 4-CPU pod at 4 torch threads: 19.9ms p50 / 46 req/s at 1 pass in
        # flight, 194.4ms p50 / 20.7 req/s at 4.
        cpus = _cgroup_cpu_limit() or os.cpu_count() or 1
        self._embed_slots = asyncio.Semaphore(
            max(1, int(cpus) // torch.get_num_threads()))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            # In K8s when the host Node machine has GPU , but the pod is not allowed to use
            # it we get assertion error.
            try:
                self.model = AutoModel.from_pretrained(model_path).cuda(0)
            except:

                self.model = AutoModel.from_pretrained(model_path)
        else:
            self.model = AutoModel.from_pretrained(model_path)
        if backend == 'redis':
            from src.utils.SAPRedis import RedisMemory
            self.storage_client = RedisMemory(
                **connection_config
            )
        elif backend == "qdrant":
            from src.utils.SAPQdrant import SAPQdrant
            self.storage_client = SAPQdrant(
                **connection_config
            )
        else:
            raise ValueError(f"Unsupported storage backend: {backend}")

    def _embed(self, query_text):
        """CPU/GPU-bound forward pass. Runs off the event loop, see __call__."""
        # padding=True pads to the longest item in the batch (here: the query
        # itself) instead of always to max_length. A 3-token query costs 3
        # tokens, not 25.
        toks = self.tokenizer.batch_encode_plus(
            [query_text], padding=True, max_length=25, truncation=True,
            return_tensors="pt")
        if self.gpu_available:
            toks = {k: v.cuda(0) for k, v in toks.items()}
        with torch.inference_mode():
            output = self.model(**toks)
        cls_rep = output[0][:, 0, :]
        return cls_rep.cpu().numpy().tolist()[0]

    async def __call__(self, query_text, count=10, similarity="cosine", bl_type=""):
        """ Runs prediction on text"""
        # torch releases the GIL during the forward pass, but calling it inline
        # in an async def still blocks the event loop for the duration, which
        # serialises every concurrent request. Hand it to the threadpool, but
        # bounded — see _embed_slots.
        async with self._embed_slots:
            vector = await run_in_threadpool(self._embed, query_text)
        logger.info(f"Calculated Vector of {len(vector)} dims,")
        return await self.storage_client.search(
            query_vector=vector,
            top_n=count,
            bl_type=bl_type,
            algorithm=similarity
        )





class ModelFactory:
    # this is populated by calling load_model
    # it stores instances
    models = {}
    # register classes here
    # when defining wrapper please register here
    # when using the config the class to be used has to be registered here.
    model_classes = {
        "TokenClassificationWrapper": TokenClassificationModelWrapper,
        'SapbertWrapper': SapbertModelWrapper
    }

    def __init__(self):
        pass

    @staticmethod
    def load_model(name, path, model_class, extra_params=None):
        if name in ModelFactory.models.keys():
            logger.info("Model %s already in class skipping initialization",
                        name)
            return
        else:
            logger.info(f"Initializing model {name}")
            assert issubclass(model_class, ModelWrapper), "Error please provide a subclass type of ModelWrapper"
            # initializes model and makes its prediction a callable
            if extra_params:
                ModelFactory.models[name] = model_class(path, **extra_params)
            else:
                ModelFactory.models[name] = model_class(path)

    @staticmethod
    async def query_model(model_name, query_text, query_count=1, **kwargs):
        if model_name not in ModelFactory.models:
            raise ModelNotFoundError(f"Model {model_name} not found")
        # since we have model as a callable class we can just treat it like a function
        return await ModelFactory.models[model_name](query_text, query_count, **kwargs)

    @staticmethod
    def get_model_names():
        return list(ModelFactory.models.keys())


def init_models(config_file_path):
    """
    Initializes models based on configuration
    :param config_file_path:
    :return:
    """
    with open(config_file_path) as config_stream:
        config = yaml.load(config_stream, Loader=yaml.SafeLoader)
    logger.info(config)
    for model_name in config:
        logger.info(model_name)
        cls = ModelFactory.model_classes.get(config[model_name]['class'], None)
        if cls is None:
            raise ValueError(
                f"model class {config[model_name]['class']} not found please use one of {ModelFactory.model_classes.keys()}, "
                f"Or add your wrapper to ModelFactory.model_classes dictionary")

        path = config[model_name]['path']
        extra_params = None

        if model_name == 'sapbert':
            extra_params = {"connection_config": config[model_name].get('connectionParams', None),
                            "backend": config[model_name]["storage"]}

        ModelFactory.load_model(name=model_name, path=path, model_class=cls, extra_params=extra_params)

        logger.info(f"Loaded {cls} model from {path} as {model_name}")


def run_main():
    "pulling this into a function keeps the top-level namespace cleaner"
    model_path = "/models/medmentions-v0.1.nemo"
    ModelFactory.load_model('medmentions', path=model_path, model_class=TokenClassificationModelWrapper)
    text = """Scientific fraud: the McBride case--judgment. Dr W G McBride, who was a specialist obstetrician and gynaecologist and the first to publish on the teratogenicity of thalidomide, has been removed from the medical register after a four-year inquiry by the Medical Tribunal of New South Wales. Of the 44 medical practice allegations made against him by the Department of Health only one minor one was found proved but 24 of the medical research allegations were found proved. Of these latter, the most serious was that in 1982 he published a scientific journal, spurious results relating to laboratory experiments on pregnant rabbits dosed with scopolamine. Had Dr McBride used any of the many opportunities available to him to make an honest disclosure of his misdemeanour, his conduct would have been excused by the Tribunal. However, he persisted in denying his fraudulent conduct for several years, including the four years of the Inquiry. The Tribunal unanimously found Dr McBride not of good character in the context of fitness to practice medicine. The decision to deregister was taken by a majority of 3 to 1. Since research science is not organized as a profession, there are no formal sanctions which can be taken against his still engaging in such research. Scientific fraud: the McBride case--judgment. Aflatoxin exposure in Singapore: blood aflatoxin levels in normal subjects, hepatitis B virus carriers and primary hepatocellular carcinoma patients. Blood screening conducted on Singaporeans over 1991-1992 showed exposure to predominantly aflatoxin B1 and to a lesser extent G1. The extent of exposure to B1 among three groups of residents in Singapore, namely normal subjects (n = 423), hepatitis B virus carriers (n = 302) and primary hepatocellular carcinoma (PHC) patients (n = 58) were extensive as reflected by the positive rates of 15.1, 0.7 and 1.7 per cent respectively. However, the degree of individual exposure to this toxin among the three groups was considered low as shown by the low respective mean blood levels of 5.4 +/- 3.2 (range 3.0-17), 7.7 (range 7.5-7.9) and 7.5 picogrammes per ml of blood. It is not immediately clear whether or not such low levels would precipitate an undesirable health effect. The higher positive rate seen in normal subjects as compared with the other groups could be due to differences in dietary intake of aflatoxin B1, differences in metabolic patterns or both. About 70 per cent of PHC patients studied were carriers. The degree of aflatoxin B1 exposure among normal subjects in Singapore was a factor of 22.1 times less than that in Japan, 40.9 times less than that in Indonesia and 51.3 times less than that in the Philippines. Similarly, the extent of exposure among hepatitis B carriers in Singapore was a factor of 8.2 times, 39.6 times and 24.2 times less than those in the other three Asiatic countries respectively. The results reflected stringent Government control over the quality of food stuff imported into this country. As Singapore imports almost all of its dietary needs from elsewhere, it can afford to be selective at a cost. Aflatoxin M1, a metabolite of B1, was most commonly encountered in the liver tissues of deceased (n = 154) who died of causes other than sickness or disease in 1992-93, consistent with our blood findings of prevalence of aflatoxin B1. High performance liquid chromatography (HPLC) with fluorescence detection using one of the aflatoxins G2 or B2 as an internal standard was used for the detection and quantification of aflatoxins. The use of an internal standard structurally and chemically similar to those required to be quantified minimizes errors in quantifications. This is because differences in the quenching of fluorescence between specimen extracts and spiked-standard extracts were internally standardized and compensated for. The presence of an internal standard also helped to locate aflatoxins of interest more accurately.(ABSTRACT TRUNCATED AT 400 WORDS)"""
    result = ModelFactory.query_model('medmentions', text)
    print(result)

# test this factory by setting the model path
if __name__ == '__main__':
    run_main()

