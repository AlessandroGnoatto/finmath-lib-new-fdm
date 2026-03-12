package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.modelling.products.CallOrPut;

/**
 * Boundary conditions for {@link EuropeanOption} under the
 * {@link FDMHestonModel}.
 *
 * <p>
 * This class provides Dirichlet-type boundary values for the
 * finite difference approximation of the Heston PDE.
 * </p>
 *
 * <p>
 * Convention for returned arrays in 2D:
 * </p>
 * <ul>
 *   <li>{@code result[0]}: boundary value for the S-dimension</li>
 *   <li>{@code result[1]}: boundary value for the v-dimension</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class EuropeanOptionHestonModelBoundary implements FiniteDifferenceBoundary {

	private final FDMHestonModel model;

	/**
	 * Creates the boundary condition associated with a given
	 * {@link FDMHestonModel}.
	 *
	 * @param model The Heston model providing risk-free and dividend curves.
	 */
	public EuropeanOptionHestonModelBoundary(final FDMHestonModel model) {
		this.model = model;
	}

	@Override
	public double[] getValueAtLowerBoundary(
			final FiniteDifferenceProduct product,
			final double time,
			final double... riskFactors) {

		final EuropeanOption option = (EuropeanOption) product;
		final CallOrPut callOrPut = option.getCallOrPut();
		final double strike = option.getStrike();
		final double maturity = option.getMaturity();

		if(riskFactors == null || riskFactors.length < 1) {
			throw new IllegalArgumentException(
					"Heston boundary requires at least S as risk factor.");
		}
		final double S = riskFactors[0];

		double t = time;
		if(t == 0) {
			t = 0.000001;
		}

		final double r =
				-Math.log(model.getRiskFreeCurve().getDiscountFactor(t)) / t;
		final double q =
				-Math.log(model.getDividendYieldCurve().getDiscountFactor(t)) / t;

		final double discountR =
				Math.exp(-r * (maturity - time));
		final double discountQ =
				Math.exp(-q * (maturity - time));

		final double[] result = new double[2];

		// S -> 0
		if(callOrPut == CallOrPut.CALL) {
			result[0] = 0.0;
		}
		else {
			result[0] = strike * discountR;
		}

		// v -> 0 (deterministic limit)
		if(callOrPut == CallOrPut.CALL) {
			result[1] =
					Math.max(
							S * discountQ - strike * discountR,
							0.0);
		}
		else {
			result[1] =
					Math.max(
							strike * discountR - S * discountQ,
							0.0);
		}

		return result;
	}

	@Override
	public double[] getValueAtUpperBoundary(
			final FiniteDifferenceProduct product,
			final double time,
			final double... riskFactors) {

		final EuropeanOption option = (EuropeanOption) product;
		final CallOrPut callOrPut = option.getCallOrPut();
		final double strike = option.getStrike();
		final double maturity = option.getMaturity();

		if(riskFactors == null || riskFactors.length < 1) {
			throw new IllegalArgumentException(
					"Heston boundary requires at least S as risk factor.");
		}
		final double S = riskFactors[0];

		double t = time;
		if(t == 0) {
			t = 0.000001;
		}

		final double r =
				-Math.log(model.getRiskFreeCurve().getDiscountFactor(t)) / t;
		final double q =
				-Math.log(model.getDividendYieldCurve().getDiscountFactor(t)) / t;

		final double discountR =
				Math.exp(-r * (maturity - time));
		final double discountQ =
				Math.exp(-q * (maturity - time));

		final double[] result = new double[2];

		// S -> infinity
		if(callOrPut == CallOrPut.CALL) {
			result[0] = S * discountQ - strike * discountR;
		}
		else {
			result[0] = 0.0;
		}

		// v -> infinity
		if(callOrPut == CallOrPut.CALL) {
			result[1] = S * discountQ;
		}
		else {
			result[1] = strike * discountR;
		}

		return result;
	}
}