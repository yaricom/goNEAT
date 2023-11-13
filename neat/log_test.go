package neat

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestAcceptLogLevel_Error(t *testing.T) {
	assert.False(t, acceptLogLevel(LogLevelError, LogLevelDebug))
	assert.False(t, acceptLogLevel(LogLevelError, LogLevelInfo))
	assert.False(t, acceptLogLevel(LogLevelError, LogLevelWarning))
	assert.True(t, acceptLogLevel(LogLevelError, LogLevelError))
}

func TestAcceptLogLevel_Warning(t *testing.T) {
	assert.False(t, acceptLogLevel(LogLevelWarning, LogLevelDebug))
	assert.False(t, acceptLogLevel(LogLevelWarning, LogLevelInfo))
	assert.True(t, acceptLogLevel(LogLevelWarning, LogLevelWarning))
	assert.True(t, acceptLogLevel(LogLevelWarning, LogLevelError))
}

func TestAcceptLogLevel_Info(t *testing.T) {
	assert.False(t, acceptLogLevel(LogLevelInfo, LogLevelDebug))
	assert.True(t, acceptLogLevel(LogLevelInfo, LogLevelInfo))
	assert.True(t, acceptLogLevel(LogLevelInfo, LogLevelWarning))
	assert.True(t, acceptLogLevel(LogLevelInfo, LogLevelError))
}

func TestAcceptLogLevel_Debug(t *testing.T) {
	assert.True(t, acceptLogLevel(LogLevelDebug, LogLevelDebug))
	assert.True(t, acceptLogLevel(LogLevelDebug, LogLevelInfo))
	assert.True(t, acceptLogLevel(LogLevelDebug, LogLevelWarning))
	assert.True(t, acceptLogLevel(LogLevelDebug, LogLevelError))
}

func TestAcceptLogLevel_Unsupported(t *testing.T) {
	assert.False(t, acceptLogLevel("unsupported", LogLevelDebug))
	assert.False(t, acceptLogLevel("unsupported", LogLevelInfo))
	assert.False(t, acceptLogLevel("unsupported", LogLevelWarning))
	assert.False(t, acceptLogLevel("unsupported", LogLevelError))
}
