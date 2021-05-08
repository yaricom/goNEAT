# Changelog

## v2.8.1
*May 8, 2021*

This release is an important milestone to finalize transition of the library to the v2.

All planed v2 milestone enhancements implemented, including but not limited to:
* support for Numpy NPZ file format for collected data samples
* changed experiment execution context to run standard GO context with NEAT options encapsulated
* implemented support for system termination signals to gracefully stop experiment execution
* fixed YAML serialization/deserialization
* fixed all unit tests to support testify and added more unit tests
* multiple bug fixed and naming convention fixes in the source code
* added Makefile with standard targets to build, run, and test as well as to run all defined experiments