package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.solvers.TwoStateActiveBoundaryProvider2D;
import net.finmath.modelling.products.CallOrPut;

/**
 * Active-state outer boundary provider for the direct two-state Heston knock-in solver.
 *
 * <p>
 * The active state represents the post-activation value, i.e. the corresponding
 * vanilla European option after the barrier has been hit.
 * </p>
 *
 * <p>
 * Conventions:
 * </p>
 * <ul>
 *   <li>State variable 0 is spot S.</li>
 *   <li>State variable 1 is variance v.</li>
 *   <li>Spot-lower / spot-upper boundaries use vanilla asymptotics.</li>
 *   <li>Variance boundaries are currently left untouched by returning {@code Double.NaN}.</li>
 * </ul>
 */
public class HestonActiveBoundaryProvider implements TwoStateActiveBoundaryProvider2D {

	private static final double EPSILON = 1E-10;

	private final FDMHestonModel model;
	private final double strike;
	private final double maturity;
	private final CallOrPut callOrPut;

	public HestonActiveBoundaryProvider(
			final FDMHestonModel model,
			final double strike,
			final double maturity,
			final CallOrPut callOrPut) {
		this.model = model;
		this.strike = strike;
		this.maturity = maturity;
		this.callOrPut = callOrPut;
	}

	@Override
	public double[] getLowerBoundaryValues(final double time, final double... stateVariables) {
		final double t = Math.max(time, EPSILON);

		final double discountFactorRiskFree = model.getRiskFreeCurve().getDiscountFactor(t);
		final double riskFreeRate = -Math.log(discountFactorRiskFree) / t;

		/*
		 * lower boundary of state variable 0 = spot lower boundary
		 * lower boundary of state variable 1 = variance lower boundary
		 *
		 * We inject only the spot boundary here.
		 */
		final double sLowerValue;
		if(callOrPut == CallOrPut.CALL) {
			sLowerValue = 0.0;
		}
		else {
			sLowerValue = strike * Math.exp(-riskFreeRate * (maturity - time));
		}

		return new double[] { sLowerValue, Double.NaN };
	}

	@Override
	public double[] getUpperBoundaryValues(final double time, final double... stateVariables) {
		final double t = Math.max(time, EPSILON);

		final double discountFactorRiskFree = model.getRiskFreeCurve().getDiscountFactor(t);
		final double riskFreeRate = -Math.log(discountFactorRiskFree) / t;

		final double discountFactorDividend = model.getDividendYieldCurve().getDiscountFactor(t);
		final double dividendYieldRate = -Math.log(discountFactorDividend) / t;

		final double spot = stateVariables[0];

		/*
		 * upper boundary of state variable 0 = spot upper boundary
		 * upper boundary of state variable 1 = variance upper boundary
		 *
		 * We inject only the spot boundary here.
		 */
		final double sUpperValue;
		if(callOrPut == CallOrPut.CALL) {
			sUpperValue =
					spot * Math.exp(-dividendYieldRate * (maturity - time))
					- strike * Math.exp(-riskFreeRate * (maturity - time));
		}
		else {
			sUpperValue = 0.0;
		}

		return new double[] { sUpperValue, Double.NaN };
	}
}