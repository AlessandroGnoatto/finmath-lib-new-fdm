package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBachelierModel;
import net.finmath.finitedifference.assetderivativevaluation.products.BarrierOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;

/**
 * Boundary conditions for {@link BarrierOption} under the {@link FDMBachelierModel}.
 *
 * <p>
 * This implementation is intended for the original one-dimensional theta solver,
 * which expects finite Dirichlet values on both sides of the spatial domain.
 * </p>
 *
 * <p>
 * Conventions:
 * </p>
 * <ul>
 *   <li>For knock-out options, the rebate is paid at hit.</li>
 *   <li>For knock-in options, the product is priced by parity in {@link BarrierOption},
 *       so this boundary class mainly matters for knock-out options.</li>
 *   <li>The barrier is assumed to coincide with one spatial boundary of the grid:
 *       down barriers at the lower boundary, up barriers at the upper boundary.</li>
 * </ul>
 *
 * <p>
 * Therefore:
 * </p>
 * <ul>
 *   <li>at the barrier boundary of an OUT option, the value is the rebate itself,</li>
 *   <li>at the opposite far boundary, the standard Bachelier vanilla asymptotic
 *       boundary condition is used.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionBachelierModelBoundary implements FiniteDifferenceBoundary {

	private static final double EPSILON = 1E-6;

	private final FDMBachelierModel model;

	/**
	 * Creates the boundary condition associated with a given
	 * {@link FDMBachelierModel}.
	 *
	 * @param model The Bachelier finite difference model.
	 */
	public BarrierOptionBachelierModelBoundary(final FDMBachelierModel model) {
		this.model = model;
	}

	@Override
	public double[] getValueAtLowerBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... riskFactors) {

		final double[] result = new double[1];
		final BarrierOption option = (BarrierOption) product;
		final BarrierType barrierType = option.getBarrierType();
		final CallOrPut sign = option.getCallOrPut();

		/*
		 * Down-and-out: lower boundary is the barrier.
		 * Rebate is paid at hit, hence no discounting here.
		 */
		if(barrierType == BarrierType.DOWN_OUT) {
			result[0] = option.getRebate();
			return result;
		}

		/*
		 * Otherwise use vanilla lower-boundary asymptotics.
		 */
		if(sign == CallOrPut.CALL) {
			result[0] = 0.0;
		}
		else {
			if(time == 0.0) {
				time = EPSILON;
			}

			final double discountFactorRiskFree =
					model.getRiskFreeCurve().getDiscountFactor(time);
			final double riskFreeRate =
					-Math.log(discountFactorRiskFree) / time;

			final double strike = option.getStrike();
			final double maturity = option.getMaturity();

			result[0] =
					strike * Math.exp(-riskFreeRate * (maturity - time));
		}

		return result;
	}

	@Override
	public double[] getValueAtUpperBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... riskFactors) {

		final double[] result = new double[1];
		final BarrierOption option = (BarrierOption) product;
		final BarrierType barrierType = option.getBarrierType();
		final CallOrPut sign = option.getCallOrPut();

		/*
		 * Up-and-out: upper boundary is the barrier.
		 * Rebate is paid at hit, hence no discounting here.
		 */
		if(barrierType == BarrierType.UP_OUT) {
			result[0] = option.getRebate();
			return result;
		}

		/*
		 * Otherwise use vanilla upper-boundary asymptotics.
		 */
		if(sign == CallOrPut.CALL) {
			if(time == 0.0) {
				time = EPSILON;
			}

			final double discountFactorRiskFree =
					model.getRiskFreeCurve().getDiscountFactor(time);
			final double riskFreeRate =
					-Math.log(discountFactorRiskFree) / time;

			final double maturity = option.getMaturity();

			result[0] =
					riskFactors[0]
					- option.getStrike() * Math.exp(-riskFreeRate * (maturity - time));
		}
		else {
			result[0] = 0.0;
		}

		return result;
	}
}