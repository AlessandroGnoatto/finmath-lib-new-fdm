package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMCevModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.modelling.products.CallOrPut;

/**
 * Boundary conditions for European options under the CEV model.
 *
 * <p>
 * The boundary logic mirrors the Black-Scholes boundary classes already present in the project:
 * discounted intrinsic/asymptotic behavior is used to provide Dirichlet-type boundary values.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class EuropeanOptionCevModelBoundary implements FiniteDifferenceBoundary {

	private final FDMCevModel model;

	/**
	 * Creates the boundary provider for the given model.
	 *
	 * @param model The CEV model instance.
	 */
	public EuropeanOptionCevModelBoundary(final FDMCevModel model) {
		this.model = model;
	}

	@Override
	public double[] getValueAtLowerBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... riskFactors) {

		final double[] result = new double[1];
		final CallOrPut sign = ((EuropeanOption) product).getCallOrPut();

		if(sign == CallOrPut.CALL) {
			result[0] = 0.0;
			return result;
		}

		// Put
		if(time == 0.0) {
			time = 1e-6;
		}

		final double rF = model.getRiskFreeCurve().getDiscountFactor(time);
		final double riskFreeRate = -Math.log(rF) / 1.0;

		final double strike = ((EuropeanOption) product).getStrike();
		final double maturity = ((EuropeanOption) product).getMaturity();

		result[0] = strike * Math.exp(-riskFreeRate * (maturity - time));
		return result;
	}

	@Override
	public double[] getValueAtUpperBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... riskFactors) {

		final double[] result = new double[1];
		final CallOrPut sign = ((EuropeanOption) product).getCallOrPut();

		if(sign == CallOrPut.PUT) {
			result[0] = 0.0;
			return result;
		}

		// Call
		if(time == 0.0) {
			time = 1e-6;
		}

		final double rF = model.getRiskFreeCurve().getDiscountFactor(time);
		final double riskFreeRate = -Math.log(rF) / time;

		final double dY = model.getDividendYieldCurve().getDiscountFactor(time);
		final double dividendYieldRate = -Math.log(dY) / time;

		final double strike = ((EuropeanOption) product).getStrike();
		final double maturity = ((EuropeanOption) product).getMaturity();

		final double dividendAdjustedStockPrice =
				riskFactors[0] * Math.exp(-dividendYieldRate * (maturity - time));

		result[0] = dividendAdjustedStockPrice - strike * Math.exp(-riskFreeRate * (maturity - time));
		return result;
	}
}