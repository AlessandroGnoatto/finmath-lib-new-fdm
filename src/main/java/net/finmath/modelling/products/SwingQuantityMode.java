package net.finmath.modelling.products;

/**
 * Quantity-control mode for a swing contract.
 *
 * <p>
 * {@link #BANG_BANG} restricts each decision to the local minimum or local maximum quantity.
 * {@link #DISCRETE_QUANTITY_GRID} allows intermediate quantities on a discretized local grid.
 * </p>
 */
public enum SwingQuantityMode {
	BANG_BANG,
	DISCRETE_QUANTITY_GRID
}