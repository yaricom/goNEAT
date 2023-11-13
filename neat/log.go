package neat

import (
	"fmt"
	"github.com/pkg/errors"
	"log"
	"os"
)

// LoggerLevel type to specify logger output level
type LoggerLevel string

const (
	// LogLevelDebug The Debug log level
	LogLevelDebug LoggerLevel = "debug"
	// LogLevelInfo The Info log level
	LogLevelInfo LoggerLevel = "info"
	// LogLevelWarning The Warning log level
	LogLevelWarning LoggerLevel = "warn"
	// LogLevelError The Error log level
	LogLevelError LoggerLevel = "error"
)

var (
	// LogLevel The current log level of the context
	LogLevel LoggerLevel

	loggerDebug = log.New(os.Stdout, "DEBUG: ", log.Ltime|log.Lshortfile)
	loggerInfo  = log.New(os.Stdout, "INFO: ", log.Ltime|log.Lshortfile)
	loggerWarn  = log.New(os.Stdout, "ALERT: ", log.Ltime|log.Lshortfile)
	loggerError = log.New(os.Stderr, "ERROR: ", log.Ltime|log.Lshortfile)

	// DebugLog The logger to output all messages
	DebugLog = func(message string) {
		if acceptLogLevel(LogLevel, LogLevelDebug) {
			_ = loggerDebug.Output(2, message)
		}
	}
	// InfoLog The logger to output messages with Info and up level
	InfoLog = func(message string) {
		if acceptLogLevel(LogLevel, LogLevelInfo) {
			_ = loggerInfo.Output(2, message)
		}
	}
	// WarnLog The logger to output messages with Warn and up level
	WarnLog = func(message string) {
		if acceptLogLevel(LogLevel, LogLevelWarning) {
			_ = loggerWarn.Output(2, message)
		}
	}
	// ErrorLog The logger to output messages with Error and up level
	ErrorLog = func(message string) {
		if acceptLogLevel(LogLevel, LogLevelError) {
			_ = loggerError.Output(2, message)
		}
	}
)

// InitLogger is to initialize logger
func InitLogger(level string) error {
	switch level {
	case "debug":
		LogLevel = LogLevelDebug
	case "info":
		LogLevel = LogLevelInfo
	case "warn":
		LogLevel = LogLevelWarning
	case "error":
		LogLevel = LogLevelError
	default:
		return errors.Errorf("unsupported log level: [%s]", level)
	}
	return nil
}

func acceptLogLevel(currentLevel, targetLevel LoggerLevel) bool {
	if currentLevel == LogLevelDebug {
		return targetLevel == LogLevelDebug || targetLevel == LogLevelInfo || targetLevel == LogLevelWarning || targetLevel == LogLevelError
	} else if currentLevel == LogLevelInfo {
		return targetLevel == LogLevelInfo || targetLevel == LogLevelWarning || targetLevel == LogLevelError
	} else if currentLevel == LogLevelWarning {
		return targetLevel == LogLevelWarning || targetLevel == LogLevelError
	} else if currentLevel == LogLevelError {
		return targetLevel == LogLevelError
	}
	_ = loggerError.Output(2, fmt.Sprintf("Unsupported NEAT log level was set: '%s'. Please use one of the following: 'debug', 'info', 'warn', and 'error'.", currentLevel))
	return false
}
