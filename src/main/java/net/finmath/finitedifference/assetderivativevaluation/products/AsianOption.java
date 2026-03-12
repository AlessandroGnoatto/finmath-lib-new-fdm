package net.finmath.finitedifference.assetderivativevaluation.products;

import java.util.function.DoubleBinaryOperator;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.boundaries.FiniteDifferenceBoundary;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.finitedifference.solvers.FDMThetaMethod2DStateVariableForm;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.modelling.products.CallOrPut;

/**
 * Fixed-strike arithmetic Asian option priced via a Markov lift.
 *
 * <p>We lift the 1D Black-Scholes state S to the 2D Markov state (S, I) where
 * I(t) = integral_0^t S(u) du, and solve a 2D PDE.</p>
 *
 * <p>Payoff at maturity T:
 * Call: max(I(T)/T - K, 0)
 * Put : max(K - I(T)/T, 0)</p>
 *
 * <p>Assumption: averaging times coincide with the PDE grid's time discretization.</p>
 */
public class AsianOption implements FiniteDifferenceProduct {

	private final String underlyingName;
	private final double maturity;
	private final double strike;
	private final CallOrPut callOrPutSign;
	private final ExerciseType exercise;

	public AsianOption(final String underlyingName, final double maturity, final double strike, final double callOrPutSign) {
		this.underlyingName = underlyingName;
		this.maturity = maturity;
		this.strike = strike;

		if(callOrPutSign == 1.0) {
			this.callOrPutSign = CallOrPut.CALL;
		}
		else if(callOrPutSign == -1.0) {
			this.callOrPutSign = CallOrPut.PUT;
		}
		else {
			throw new IllegalArgumentException("Unknown option type.");
		}

		this.exercise = ExerciseType.EUROPEAN;
	}

	public AsianOption(final String underlyingName, final double maturity, final double strike, final CallOrPut callOrPutSign) {
		this.underlyingName = underlyingName;
		this.maturity = maturity;
		this.strike = strike;
		this.callOrPutSign = callOrPutSign;
		this.exercise = ExerciseType.EUROPEAN;
	}

	public AsianOption(final double maturity, final double strike, final double callOrPutSign) {
		this(null, maturity, strike, callOrPutSign);
	}

	public AsianOption(final double maturity, final double strike, final CallOrPut callOrPutSign) {
		this(null, maturity, strike, callOrPutSign);
	}

	public AsianOption(final String underlyingName, final double maturity, final double strike) {
		this(underlyingName, maturity, strike, CallOrPut.CALL);
	}

	public AsianOption(final double maturity, final double strike) {
		this(null, maturity, strike, CallOrPut.CALL);
	}

	@Override
	public double[] getValue(final double evaluationTime, final FiniteDifferenceEquityModel model) {
		final double[][] values = getValues(model);

		final double tau = maturity - evaluationTime;
		final int timeIndex = model.getSpaceTimeDiscretization()
				.getTimeDiscretization()
				.getTimeIndexNearestLessOrEqual(tau);

		final double[] column = new double[values.length];
		for(int i = 0; i < values.length; i++) {
			column[i] = values[i][timeIndex];
		}
		return column;
	}

	@Override
	public double[][] getValues(final FiniteDifferenceEquityModel model) {

		if(!(model instanceof FDMBlackScholesModel)) {
			throw new IllegalArgumentException("AsianOption currently supports only FDMBlackScholesModel.");
		}

		final FDMBlackScholesModel bsModel = (FDMBlackScholesModel) model;

		final SpaceTimeDiscretization baseDiscretization = bsModel.getSpaceTimeDiscretization();
		final Grid sGrid = baseDiscretization.getSpaceGrid(0);

		final double[] sNodes = sGrid.getGrid();
		final double sMax = sNodes[sNodes.length - 1];
		final double iMax = sMax;

		final int nI = sNodes.length;
		final Grid iGrid = new UniformGrid(nI - 1, 0.0, iMax);

		final SpaceTimeDiscretization liftedDiscretization = new SpaceTimeDiscretization(
				new Grid[] { sGrid, iGrid },
				baseDiscretization.getTimeDiscretization(),
				baseDiscretization.getTheta(),
				new double[] { bsModel.getInitialValue(), 0.0 }
		);

		final FiniteDifferenceEquityModel liftedModel =
				new LiftedFDMBlackScholesModelDecorator(bsModel, liftedDiscretization);

		final FDMThetaMethod2DStateVariableForm solver =
				new FDMThetaMethod2DStateVariableForm(liftedModel, this, liftedDiscretization, exercise);

		final DoubleBinaryOperator payoffAtMaturity = (S, I) -> {
			final double average = I / maturity;
			if(callOrPutSign == CallOrPut.CALL) {
				return Math.max(average - strike, 0.0);
			}
			else {
				return Math.max(strike - average, 0.0);
			}
		};

		// Uses the new overload in FDMThetaMethod2D
		return solver.getValues(maturity, payoffAtMaturity);
	}

	public String getUnderlyingName() {
		return underlyingName;
	}

	public double getMaturity() {
		return maturity;
	}

	public double getStrike() {
		return strike;
	}

	public CallOrPut getCallOrPut() {
		return callOrPutSign;
	}

	public ExerciseType getExercise() {
		return exercise;
	}

	/**
	 * Decorator that lifts a 1D Black-Scholes model to a 2D model with state (S, I),
	 * where I(t) = integral_0^t S(u) du.
	 */
	private static final class LiftedFDMBlackScholesModelDecorator implements FiniteDifferenceEquityModel, FiniteDifferenceBoundary {

		private final FDMBlackScholesModel delegate;
		private final SpaceTimeDiscretization liftedDiscretization;

		private LiftedFDMBlackScholesModelDecorator(final FDMBlackScholesModel delegate, final SpaceTimeDiscretization liftedDiscretization) {
			this.delegate = delegate;
			this.liftedDiscretization = liftedDiscretization;
		}

		@Override
		public DiscountCurve getRiskFreeCurve() {
			return delegate.getRiskFreeCurve();
		}

		@Override
		public DiscountCurve getDividendYieldCurve() {
			return delegate.getDividendYieldCurve();
		}

		@Override
		public SpaceTimeDiscretization getSpaceTimeDiscretization() {
			return liftedDiscretization;
		}

		@Override
		public double[] getDrift(double time, final double... stateVariables) {
			if(time == 0.0) {
				time = 1e-6;
			}

			final double S = stateVariables.length > 0 ? stateVariables[0] : delegate.getInitialValue();
			final double I = stateVariables.length > 0 ? stateVariables[1] : delegate.getInitialValue();

			// percentage drift for S (same convention as existing BS model)
			final double muS = delegate.getDrift(time, S)[0];

			// drift for I: dI = S dt
			final double muI = S;

			return new double[] { muS, muI };
		}

		@Override
		public double[][] getFactorLoading(final double time, final double... stateVariables) {
			final double S = stateVariables.length > 0 ? stateVariables[0] : delegate.getInitialValue();

			// percentage loading for S from base BS model
			final double sigma = delegate.getFactorLoading(time, S)[0][0];

			// I has no diffusion
			return new double[][] { { sigma, 0.0 }, { 0.0, 0.0 } };
		}

		@Override
		public double[] getValueAtLowerBoundary(
				final FiniteDifferenceProduct product,
				double time,
				final double... riskFactors) {

			if(time == 0.0) {
				time = 1e-6;
			}

			final AsianOption option = (AsianOption) product;
			final double maturity = option.getMaturity();
			final double strike = option.getStrike();
			final CallOrPut callOrPut = option.getCallOrPut();

			final double r = -Math.log(getRiskFreeCurve().getDiscountFactor(time)) / time;
			final double discount = Math.exp(-r * (maturity - time));

			final double[] result = new double[2];

			// S -> 0
			result[0] = (callOrPut == CallOrPut.CALL) ? 0.0 : strike * discount;

			// I -> 0 (average -> 0)
			result[1] = Double.NaN;

			return result;
		}

		@Override
		public double[] getValueAtUpperBoundary(
				final FiniteDifferenceProduct product,
				double time,
				final double... riskFactors) {

			if(time == 0.0) {
				time = 1e-6;
			}

			final AsianOption option = (AsianOption) product;
			final double maturity = option.getMaturity();
			final double strike = option.getStrike();
			final CallOrPut callOrPut = option.getCallOrPut();

			final double r = -Math.log(getRiskFreeCurve().getDiscountFactor(time)) / time;
			final double discount = Math.exp(-r * (maturity - time));

			final double S = riskFactors.length > 0 ? riskFactors[0] : 0.0;
			final double I = riskFactors.length > 1 ? riskFactors[1] : 0.0;

			final double[] result = new double[2];

			// S -> infinity (simple asymptotic bound, consistent with vanilla style)
			result[0] = (callOrPut == CallOrPut.CALL) ? (S - strike * discount) : 0.0;

			// I -> infinity: payoff dominated by average = I/T
			final double average = I / maturity;
			final double intrinsic = (callOrPut == CallOrPut.CALL)
					? Math.max(average - strike, 0.0)
					: Math.max(strike - average, 0.0);

			result[1] = intrinsic * discount;

			return result;
		}
	}
}