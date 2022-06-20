# Integration tests

These integration tests are intended to test a fully-functional Nemo-Serve
instance. While these tests default to https://med-nemo.apps.renci.org/,
but you can change this by setting the `NEMOSERVE_URL` environmental variable. 

You can run these tests by running `pytest` from the root directory of this
project or `pytest -s tests/integration` if you only want to run the
integration tests.
