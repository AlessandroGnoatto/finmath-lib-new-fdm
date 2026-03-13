package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBachelierModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.modelling.products.CallOrPut;

/**
 * Boundary conditions for {@link EuropeanOption} under the {@link FDMBachelierModel}.
 *
 * <p>
 * This class provides Dirichlet-type boundary values for the finite difference approximation of a
 * Bachelier (normal) model PDE.
 * </p>
 *
 * <p>
 * The asymptotic behavior for large absolute underlying values is linear in the underlying:
 * for calls, the upper boundary behaves like the discounted forward intrinsic;
 * for puts, the lower boundary behaves like the discounted forward intrinsic.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class EuropeanOptionBachelierModelBoundary implements FiniteDifferenceBoundary {

	private static final double EPSILON = 1E-6;

	private final FDMBachelierModel model;

	/**
	 * Creates the boundary condition associated with a given Bachelier model.
	 *
	 * @param model The Bachelier model providing risk-free and dividend curves.
	 */
	public EuropeanOptionBachelierModelBoundary(final FDMBachelierModel model) {
		this.model = model;
	}

	@Override
	public double[] getValueAtLowerBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... riskFactors) {

		final double[] result = new double[1];
		final EuropeanOption option = (EuropeanOption) product;
		final CallOrPut sign = option.getCallOrPut();

		// x -> -infty
		if(sign == CallOrPut.CALL) {
			result[0] = 0.0;
			return result;
		}

		// Put option
		time = Math.max(time, EPSILON);

		final double discountFactorRiskFree =
				model.getRiskFreeCurve().getDiscountFactor(time);
		final double riskFreeRate =
				-Math.log(discountFactorRiskFree) / time;

		final double discountFactorDividend =
				model.getDividendYieldCurve().getDiscountFactor(time);
		final double dividendYieldRate =
				-Math.log(discountFactorDividend) / time;

		final double strike = option.getStrike();
		final double maturity = option.getMaturity();

		final double dividendAdjustedUnderlying =
				riskFactors[0] * Math.exp(-dividendYieldRate * (maturity - time));

		result[0] =
				strike * Math.exp(-riskFreeRate * (maturity - time))
				- dividendAdjustedUnderlying;

		return result;
	}

	@Override
	public double[] getValueAtUpperBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... riskFactors) {

		final double[] result = new double[1];
		final EuropeanOption option = (EuropeanOption) product;
		final CallOrPut sign = option.getCallOrPut();

		// x -> +infty
		if(sign == CallOrPut.PUT) {
			result[0] = 0.0;
			return result;
		}

		// Call option
		time = Math.max(time, EPSILON);

		final double discountFactorRiskFree =
				model.getRiskFreeCurve().getDiscountFactor(time);
		final double riskFreeRate =
				-Math.log(discountFactorRiskFree) / time;

		final double discountFactorDividend =
				model.getDividendYieldCurve().getDiscountFactor(time);
		final double dividendYieldRate =
				-Math.log(discountFactorDividend) / time;

		final double strike = option.getStrike();
		final double maturity = option.getMaturity();

		final double dividendAdjustedUnderlying =
				riskFactors[0] * Math.exp(-dividendYieldRate * (maturity - time));

		result[0] =
				dividendAdjustedUnderlying
				- strike * Math.exp(-riskFreeRate * (maturity - time));

		return result;
	}
}