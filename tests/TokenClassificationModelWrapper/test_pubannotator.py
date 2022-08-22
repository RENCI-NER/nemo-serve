"""
A tests for the PubAnnotator output format.
"""
import os

from ModelSingleton import TokenClassificationModelWrapper
from unittest.mock import Mock


def test_pubannotator():
    # Abstract from PubMed 30925593 (https://pubmed.ncbi.nlm.nih.gov/30925593/)
    # As annotated by Nemo-Serve on PMID
    pubmed_30925593_abstract = "Discovering small molecules as Wnt inhibitors that promote heart regeneration and " \
                               "injury repair There are intense interests in discovering proregenerative medicine " \
                               "leads that can promote cardiac differentiation and regeneration, as well as repair " \
                               "damaged heart tissues. We have combined zebrafish embryo-based screens with " \
                               "cardiomyogenesis assays to discover selective small molecules that modulate heart " \
                               "development and regeneration with minimal adverse effects. Two related compounds with " \
                               "novel structures, named as Cardiomogen 1 and 2 (CDMG1 and CDMG2), were identified for " \
                               "their capacity to promote myocardial hyperplasia through expansion of the cardiac " \
                               "progenitor cell population. We find that Cardiomogen acts as a Wnt inhibitor by " \
                               "targeting β-catenin and reducing Tcf/Lef-mediated transcription in cultured cells. " \
                               "CDMG treatment of amputated zebrafish hearts reduces nuclear β-catenin in injured " \
                               "heart tissue, increases cardiomyocyte (CM) proliferation, and expedites wound " \
                               "healing, thus accelerating cardiac muscle regeneration. Importantly, Cardiomogen can " \
                               "alleviate the functional deterioration of mammalian hearts after myocardial " \
                               "infarction. Injured hearts exposed to CDMG1 display increased newly formed CMs and " \
                               "reduced fibrotic scar tissue, which are in part attributable to the β-catenin " \
                               "reduction. Our findings indicate Cardiomogen as a Wnt inhibitor in enhancing " \
                               "injury-induced CM proliferation and heart regeneration, highlighting the values of " \
                               "embryo-based small molecule screens in discovery of effective and safe medicine leads. "
    pubmed_30925593_abstract_annotated = [
        "Discovering[0] small[B-biolink:ChemicalEntity] molecules[I-biolink:ChemicalEntity] as[0] Wnt["
        "B-biolink:Protein] inhibitors[B-biolink:NamedThing] that[0] promote[B-biolink:NamedThing] heart["
        "B-biolink:NamedThing] regeneration[B-biolink:NamedThing] and[0] injury[B-biolink:Disease] repair["
        "I-biolink:Disease] There[0] are[0] intense[0] interests[0] in[0] discovering[0] proregenerative["
        "B-biolink:NamedThing] medicine[B-biolink:NamedThing] leads[I-biolink:NamedThing] that[0] can[0] promote["
        "B-biolink:NamedThing] cardiac[B-biolink:NamedThing] differentiation[B-biolink:NamedThing] and[0] "
        "regeneration[B-biolink:NamedThing], as[0] well[0] as[0] repair[B-biolink:NamedThing] damaged["
        "B-biolink:NamedThing] heart[B-biolink:GrossAnatomicalStructure] tissues[I-biolink:GrossAnatomicalStructure]. "
        "We[0] have[0] combined[0] zebrafish[B-biolink:NamedThing] embryo-based[B-biolink:AnatomicalEntity] screens["
        "B-biolink:NamedThing] with[0] cardiomyogenesis[B-biolink:NamedThing] assays[B-biolink:NamedThing] to[0] "
        "discover[0] selective[0] small[B-biolink:ChemicalEntity] molecules[I-biolink:ChemicalEntity] that[0] "
        "modulate[B-biolink:NamedThing] heart[B-biolink:GrossAnatomicalStructure] development[I-biolink:NamedThing] "
        "and[0] regeneration[B-biolink:NamedThing] with[0] minimal[0] adverse[B-biolink:Disease] effects["
        "I-biolink:Disease]. Two[0] related[0] compounds[B-biolink:NamedThing] with[0] novel[B-biolink:NamedThing] "
        "structures[I-biolink:NamedThing], named[0] as[0] Cardiomogen[B-biolink:Protein] [I-biolink:Protein]1 and[0] "
        "[B-biolink:ChemicalEntity]2 (CDMG[0]1 and[0] CDMG2)[B-biolink:Protein], were[0] identified["
        "B-biolink:NamedThing] for[0] their[0] capacity[B-biolink:NamedThing] to[0] promote[B-biolink:NamedThing] "
        "myocardial[B-biolink:NamedThing] hyperplasia[I-biolink:NamedThing] through[0] expansion["
        "B-biolink:NamedThing] of[0] the[0] cardiac[B-biolink:GrossAnatomicalStructure] progenitor[B-biolink:Cell] "
        "cell[I-biolink:Cell] population[I-biolink:Cell]. We[0] find[0] that[0] Cardiomogen[B-biolink:Protein] acts["
        "0] as[0] a[0] Wnt[B-biolink:Protein] inhibitor[B-biolink:NamedThing] by[0] targeting[B-biolink:NamedThing] "
        "β-catenin[B-biolink:Protein] and[0] reducing[0] Tcf/Lef-mediated[B-biolink:Protein] transcription["
        "B-biolink:NamedThing] in[0] cultured[B-biolink:Cell] cells[I-biolink:Cell]. CDMG[B-biolink:ChemicalEntity] "
        "treatment[B-biolink:NamedThing] of[0] amputated[B-biolink:NamedThing] zebrafish[B-biolink:NamedThing] "
        "hearts[B-biolink:GrossAnatomicalStructure] reduces[B-biolink:NamedThing] nuclear[B-biolink:Protein] "
        "β-catenin[B-biolink:Protein] in[0] injured[B-biolink:Disease] heart[B-biolink:GrossAnatomicalStructure] "
        "tissue[I-biolink:AnatomicalEntity], increases[0] cardiomyocyte[B-biolink:Cell] (CM[I-biolink:NamedThing]) "
        "proliferation[B-biolink:NamedThing], and[0] expedites[0] wound[B-biolink:NamedThing] healing["
        "I-biolink:NamedThing], thus[0] accelerating[0] cardiac[B-biolink:GrossAnatomicalStructure] muscle["
        "I-biolink:NamedThing] regeneration[B-biolink:NamedThing]. Importantly[0], Cardiomogen[B-biolink:Protein] "
        "can[0] alleviate[0] the[0] functional[B-biolink:PhenotypicFeature] deterioration["
        "I-biolink:PhenotypicFeature] of[0] mammalian[B-biolink:NamedThing] hearts["
        "B-biolink:GrossAnatomicalStructure] after[0] myocardial[B-biolink:Disease] infarction[I-biolink:Disease]. "
        "Injured[B-biolink:NamedThing] hearts[B-biolink:GrossAnatomicalStructure] exposed[B-biolink:NamedThing] to["
        "I-biolink:NamedThing] CDMG[B-biolink:Protein]1 display[0] increased[0] newly[0] formed[0] CMs["
        "B-biolink:Cell] and[0] reduced[B-biolink:NamedThing] fibrotic[B-biolink:Disease] scar[B-biolink:Disease] "
        "tissue[I-biolink:Disease], which[0] are[0] in[0] part[0] attributable[0] to[0] the[0] β-catenin["
        "B-biolink:Protein] reduction[B-biolink:NamedThing]. Our[0] findings[0] indicate[0] Cardiomogen["
        "B-biolink:Protein] as[0] a[0] Wnt[B-biolink:Protein] inhibitor[B-biolink:NamedThing] in[0] enhancing["
        "B-biolink:NamedThing] injury-induced[B-biolink:Disease] CM[B-biolink:Cell] proliferation["
        "B-biolink:NamedThing] and[0] heart[B-biolink:NamedThing] regeneration[B-biolink:NamedThing], highlighting[0] "
        "the[0] values[0] of[0] embryo-based[B-biolink:AnatomicalEntity] small[B-biolink:ChemicalEntity] molecule["
        "I-biolink:ChemicalEntity] screens[B-biolink:NamedThing] in[0] discovery[B-biolink:NamedThing] of[0] "
        "effective[B-biolink:NamedThing] and[0] safe[0] medicine[B-biolink:NamedThing] leads[I-biolink:NamedThing]. "
    ]

    tcmw = TokenClassificationModelWrapper(os.environ.get("NEMO_MODEL"))

    pubannotator_output = TokenClassificationModelWrapper._pubannotate(tcmw, pubmed_30925593_abstract, pubmed_30925593_abstract_annotated)
    assert pubannotator_output == []