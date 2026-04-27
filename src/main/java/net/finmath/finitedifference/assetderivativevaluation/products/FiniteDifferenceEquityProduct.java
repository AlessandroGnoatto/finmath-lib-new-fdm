package net.finmath.finitedifference.assetderivativevaluation.products;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.modelling.Model;
import net.finmath.modelling.Product;

/**
 * Interface for products valued by a finite difference equity model.
 *
 * <p>
 * Implementations provide valuation methods compatible with
 * {@link FiniteDifferenceEquityModel}. In addition to the standard
 * {@link Product} interface, this interface exposes methods returning
 * either a single time slice or the full time-space grid of values.
 * </p>
 *
 * @author Christian Fries
 * @version 1.0
 */
public interface FiniteDifferenceEquityProduct extends Product {

	/**
	 * Returns the value of the product at a given evaluation time under
	 * the specified finite difference model.
	 *
	 * @param evaluationTime The time at which the value is evaluated (typically 0).
	 * @param model          The finite difference model used for the valuation.
	 * @return A one-dimensional array representing the option values
	 *         at the specified evaluation time.
	 */
	double[] getValue(double evaluationTime, FiniteDifferenceEquityModel model);

	/**
	 * Returns the full time-space grid of values under the specified
	 * finite difference model.
	 *
	 * @param model The finite difference model used for the valuation.
	 * @return A two-dimensional array representing the option values
	 *         over the time-space grid.
	 */
	double[][] getValues(FiniteDifferenceEquityModel model);

	@Override
	default Object getValue(final double evaluationTime, final Model model) {

		if(model instanceof FiniteDifferenceEquityModel) {
			return getValue(evaluationTime, (FiniteDifferenceEquityModel) model);
		}
		else {
			throw new IllegalArgumentException(
					"The product " + this.getClass()
					+ " cannot be valued against a model "
					+ model.getClass() + ". "
					+ "It requires a model of type "
					+ FiniteDifferenceEquityModel.class + ".");
		}
	}
}