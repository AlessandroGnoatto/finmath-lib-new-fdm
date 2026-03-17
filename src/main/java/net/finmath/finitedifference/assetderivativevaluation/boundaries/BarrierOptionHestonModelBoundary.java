package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.products.BarrierOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;

/**
 * Boundary conditions for {@link BarrierOption} under the {@link FDMHestonModel}.
 *
 * <p>
 * This implementation is intended for the finite-difference solvers which expect
 * finite Dirichlet values on the spatial boundaries.
 * </p>
 *
 * <p>
 * Conventions:
 * </p>
 * <ul>
 *   <li>For knock-out options, the rebate is paid at hit.</li>
 *   <li>For knock-in options, the product is typically priced by parity in
 *       {@link BarrierOption}, so this boundary class mainly matters for knock-out options.</li>
 *   <li>The first state variable is assumed to be the asset price.</li>
 *   <li>The barrier is assumed to coincide with one asset-price boundary of the grid:
 *       down barriers at the lower asset-price boundary, up barriers at the upper asset-price boundary.</li>
 * </ul>
 *
 * <p>
 * Therefore:
 * </p>
 * <ul>
 *   <li>at the barrier boundary of an OUT option, the value is the rebate itself,</li>
 *   <li>at the opposite far asset-price boundary, the standard Heston-style vanilla asymptotic
 *       boundary condition is used, based on risk-free and dividend discounting,</li>
 *   <li>for the variance dimension, this class returns asymptotic values consistent with the
 *       asset-price boundary formulas.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionHestonModelBoundary implements FiniteDifferenceBoundary {

	private static final double EPSILON = 1E-6;

	private final FDMHestonModel model;

	/**
	 * Creates the boundary condition associated with a given
	 * {@link FDMHestonModel}.
	 *
	 * @param model The Heston finite difference model.
	 */
	public BarrierOptionHestonModelBoundary(final FDMHestonModel model) {
		this.model = model;
	}

	@Override
	public double[] getValueAtLowerBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... riskFactors) {

		final double[] result = new double[2];
		final BarrierOption option = (BarrierOption) product;
		final BarrierType barrierType = option.getBarrierType();
		final CallOrPut sign = option.getCallOrPut();

		/*
		 * riskFactors[0] = asset price
		 * riskFactors[1] = variance
		 */
		final double stock = riskFactors.length > 0 ? riskFactors[0] : 0.0;

		if(time == 0.0) {
			time = EPSILON;
		}

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

		/*
		 * Dimension 0: lower asset-price boundary.
		 */
		if(barrierType == BarrierType.DOWN_OUT) {
			result[0] = option.getRebate();
		}
		else if(sign == CallOrPut.CALL) {
			result[0] = 0.0;
		}
		else {
			result[0] =
					strike * Math.exp(-riskFreeRate * (maturity - time));
		}

		/*
		 * Dimension 1: lower variance boundary.
		 * Use vanilla asymptotics consistent with the asset dimension.
		 */
		if(sign == CallOrPut.CALL) {
			result[1] =
					stock * Math.exp(-dividendYieldRate * (maturity - time))
					- strike * Math.exp(-riskFreeRate * (maturity - time));
		}
		else {
			result[1] = 0.0;
		}

		return result;
	}

	@Override
	public double[] getValueAtUpperBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... riskFactors) {

		final double[] result = new double[2];
		final BarrierOption option = (BarrierOption) product;
		final BarrierType barrierType = option.getBarrierType();
		final CallOrPut sign = option.getCallOrPut();

		/*
		 * riskFactors[0] = asset price
		 * riskFactors[1] = variance
		 */
		final double stock = riskFactors.length > 0 ? riskFactors[0] : 0.0;

		if(time == 0.0) {
			time = EPSILON;
		}

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

		/*
		 * Dimension 0: upper asset-price boundary.
		 */
		if(barrierType == BarrierType.UP_OUT) {
			result[0] = option.getRebate();
		}
		else if(sign == CallOrPut.CALL) {
			result[0] =
					stock * Math.exp(-dividendYieldRate * (maturity - time))
					- strike * Math.exp(-riskFreeRate * (maturity - time));
		}
		else {
			result[0] = 0.0;
		}

		/*
		 * Dimension 1: upper variance boundary.
		 * Use vanilla asymptotics consistent with the asset dimension.
		 */
		if(sign == CallOrPut.CALL) {
			result[1] =
					stock * Math.exp(-dividendYieldRate * (maturity - time))
					- strike * Math.exp(-riskFreeRate * (maturity - time));
		}
		else {
			result[1] = 0.0;
		}

		return result;
	}
}