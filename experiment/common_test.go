package experiment

import "errors"

const alwaysErrorText = "always be failing"

var alwaysError = errors.New(alwaysErrorText)

type ErrorWriter int

func (e ErrorWriter) Write(_ []byte) (int, error) {
	return 0, alwaysError
}

type ErrorReader int

func (e ErrorReader) Read(_ []byte) (n int, err error) {
	return 0, alwaysError
}
