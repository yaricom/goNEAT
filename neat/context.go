package neat

import (
	"context"
	"errors"
)

var ErrNEATOptionsNotFound = errors.New("NEAT options not found in the context")

// key is an unexported type for keys defined in this package.
// This prevents collisions with keys defined in other packages.
type key int

// neatOptionsKey is the key for neat.Options values in Contexts. It is
// unexported; clients use neat.NewContext and neat.FromContext
// instead of using this key directly.
var neatOptionsKey key

// NewContext returns a new Context that carries value of NEAT options.
func NewContext(ctx context.Context, opts *Options) context.Context {
	return context.WithValue(ctx, neatOptionsKey, opts)
}

// FromContext returns the NEAT Options value stored in ctx, if any.
func FromContext(ctx context.Context) (*Options, bool) {
	u, ok := ctx.Value(neatOptionsKey).(*Options)
	return u, ok
}
