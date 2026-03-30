package net.finmath.finitedifference.assetderivativevaluation.products;

import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;

import net.finmath.finitedifference.assetderivativevaluation.boundaries.FiniteDifferenceBoundary;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.boundaries.StandardBoundaryCondition;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.adi.FDMAsianADI2D;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.AsianStrike;
import net.finmath.modelling.products.CallOrPut;

/**
 * Arithmetic Asian option priced via a Markov lift.
 *
 * <p>
 * We lift the 1D Black-Scholes state S to the 2D Markov state (S, I) where
 * I(t) = integral_0^t S(u) du, and solve a 2D PDE.
 * </p>
 *
 * <p>
 * Fixed-strike payoff at maturity T:
 * Call: max(I(T)/T - K, 0)
 * Put : max(K - I(T)/T, 0)
 * </p>
 *
 * <p>
 * Floating-strike payoff at maturity T:
 * Call: max(S(T) - I(T)/T, 0)
 * Put : max(I(T)/T - S(T), 0)
 * </p>
 *
 * <p>
 * Assumption: averaging times coincide with the PDE grid's time discretization.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class AsianOption implements FiniteDifferenceProduct {

	private final String underlyingName;
	private final double maturity;
	private final double strike;
	private final CallOrPut callOrPutSign;
	private final AsianStrike asianStrike;
	private final Exercise exercise;

	public AsianOption(
			final String underlyingName,
			final double maturity,
			final double strike,
			final CallOrPut callOrPutSign,
			final AsianStrike asianStrike,
			final Exercise exercise) {

		if (callOrPutSign == null) {
			throw new IllegalArgumentException("Option type must not be null.");
		}
		if (asianStrike == null) {
			throw new IllegalArgumentException("Asian strike type must not be null.");
		}
		if (exercise == null) {
			throw new IllegalArgumentException("Exercise must not be null.");
		}
		if (asianStrike == AsianStrike.FIXED_STRIKE && Double.isNaN(strike)) {
			throw new IllegalArgumentException("Strike must be specified for fixed-strike Asian options.");
		}

		this.underlyingName = underlyingName;
		this.maturity = maturity;
		this.strike = strike;
		this.callOrPutSign = callOrPutSign;
		this.asianStrike = asianStrike;
		this.exercise = exercise;
	}

	public AsianOption(
			final String underlyingName,
			final double maturity,
			final double strike,
			final CallOrPut callOrPutSign) {
		this(
				underlyingName,
				maturity,
				strike,
				callOrPutSign,
				AsianStrike.FIXED_STRIKE,
				new EuropeanExercise(maturity));
	}

	public AsianOption(
			final String underlyingName,
			final double maturity,
			final double strike,
			final double callOrPutSign) {
		this(underlyingName, maturity, strike, callOrPutFromDouble(callOrPutSign));
	}

	public AsianOption(
			final double maturity,
			final double strike,
			final CallOrPut callOrPutSign) {
		this(null, maturity, strike, callOrPutSign);
	}

	public AsianOption(
			final double maturity,
			final double strike,
			final double callOrPutSign) {
		this(null, maturity, strike, callOrPutSign);
	}

	public AsianOption(
			final String underlyingName,
			final double maturity,
			final double strike) {
		this(underlyingName, maturity, strike, CallOrPut.CALL);
	}

	public AsianOption(
			final double maturity,
			final double strike) {
		this(null, maturity, strike, CallOrPut.CALL);
	}

	public AsianOption(
			final String underlyingName,
			final double maturity,
			final CallOrPut callOrPutSign,
			final AsianStrike asianStrike) {
		this(
				underlyingName,
				maturity,
				asianStrike == AsianStrike.FLOATING_STRIKE ? Double.NaN : 0.0,
				callOrPutSign,
				asianStrike,
				new EuropeanExercise(maturity));
	}

	public AsianOption(
			final double maturity,
			final CallOrPut callOrPutSign,
			final AsianStrike asianStrike) {
		this(null, maturity, callOrPutSign, asianStrike);
	}

	private static CallOrPut callOrPutFromDouble(final double callOrPutSign) {
		if (callOrPutSign == 1.0) {
			return CallOrPut.CALL;
		} else if (callOrPutSign == -1.0) {
			return CallOrPut.PUT;
		} else {
			throw new IllegalArgumentException("Unknown option type.");
		}
	}

	private double payoff(final double average, final double spot) {
		if (asianStrike == AsianStrike.FIXED_STRIKE) {
			if (callOrPutSign == CallOrPut.CALL) {
				return Math.max(average - strike, 0.0);
			} else {
				return Math.max(strike - average, 0.0);
			}
		} else if (asianStrike == AsianStrike.FLOATING_STRIKE) {
			if (callOrPutSign == CallOrPut.CALL) {
				return Math.max(spot - average, 0.0);
			} else {
				return Math.max(average - spot, 0.0);
			}
		} else {
			throw new IllegalArgumentException("Unrecognized strike type.");
		}
	}

	@Override
	public double[] getValue(final double evaluationTime, final FiniteDifferenceEquityModel model) {
		final double[][] values = getValues(model);

		final double tau = maturity - evaluationTime;
		final int timeIndex = model.getSpaceTimeDiscretization().getTimeDiscretization()
				.getTimeIndexNearestLessOrEqual(tau);

		final double[] column = new double[values.length];
		for (int i = 0; i < values.length; i++) {
			column[i] = values[i][timeIndex];
		}
		return column;
	}

	@Override
	public double[][] getValues(final FiniteDifferenceEquityModel model) {

		if (!exercise.isEuropean()) {
			throw new IllegalArgumentException("AsianOption currently supports only European exercise.");
		}

		final FiniteDifferenceEquityModel liftedModel = getLiftedModel(model);
		final FDMSolver solver = getSolver(liftedModel);

		final DoubleBinaryOperator payoffAtMaturity = (S, I) -> {
			final double average = I / maturity;
			return payoff(average, S);
		};

		return solver.getValues(maturity, payoffAtMaturity);
	}

	public FDMSolver getSolver(final FiniteDifferenceEquityModel model) {
		if (model instanceof LiftedFDMBlackScholesModelDecorator) {
			return new FDMAsianADI2D(model, this, model.getSpaceTimeDiscretization(), exercise);
		}
		throw new IllegalArgumentException("AsianOption currently supports only FDMBlackScholesModel.");
	}

	public FiniteDifferenceEquityModel getLiftedModel(final FiniteDifferenceEquityModel model) {

		final SpaceTimeDiscretization baseDiscretization = model.getSpaceTimeDiscretization();

		if (model instanceof FDMBlackScholesModel) {
			final FDMBlackScholesModel bsModel = (FDMBlackScholesModel) model;

			final Grid sGrid = baseDiscretization.getSpaceGrid(0);
			final double[] sNodes = sGrid.getGrid();
			final double sMax = sNodes[sNodes.length - 1];

			/*
			 * Since I(t) = integral_0^t S(u) du, the natural scale of I is O(T * S).
			 */
			final double iMax = maturity * sMax;

			final int nI = sNodes.length * 4;
			final Grid iGrid = new UniformGrid(nI - 1, 0.0, iMax);

			final SpaceTimeDiscretization liftedDiscretization = new SpaceTimeDiscretization(
					new Grid[] { sGrid, iGrid },
					baseDiscretization.getTimeDiscretization(),
					baseDiscretization.getTheta(),
					new double[] { bsModel.getInitialValue()[0], 0.0 });

			return new LiftedFDMBlackScholesModelDecorator(bsModel, liftedDiscretization);
		} else {
			throw new IllegalArgumentException("AsianOption currently supports only FDMBlackScholesModel.");
		}
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

	public AsianStrike getAsianStrike() {
		return asianStrike;
	}

	public Exercise getExercise() {
		return exercise;
	}

	/**
	 * Decorator that lifts a 1D Black-Scholes model to a 2D model with state (S, I),
	 * where I(t) = integral_0^t S(u) du.
	 */
	private static final class LiftedFDMBlackScholesModelDecorator
			implements FiniteDifferenceEquityModel, FiniteDifferenceBoundary {

		private final FDMBlackScholesModel delegate;
		private final SpaceTimeDiscretization liftedDiscretization;

		private LiftedFDMBlackScholesModelDecorator(
				final FDMBlackScholesModel delegate,
				final SpaceTimeDiscretization liftedDiscretization) {
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
			if (time == 0.0) {
				time = 1e-6;
			}

			final double S = stateVariables.length > 0 ? stateVariables[0] : delegate.getInitialValue()[0];

			final double muS = delegate.getDrift(time, S)[0];
			final double muI = S;

			return new double[] { muS, muI };
		}

		@Override
		public double[][] getFactorLoading(final double time, final double... stateVariables) {
			final double S = stateVariables.length > 0 ? stateVariables[0] : delegate.getInitialValue()[0];

			final double sigma = delegate.getFactorLoading(time, S)[0][0];

			return new double[][] { { sigma, 0.0 }, { 0.0, 0.0 } };
		}

		@Override
		public double[] getInitialValue() {
			final double[] oldArray = delegate.getInitialValue();
			final double[] newArray = Arrays.copyOf(oldArray, oldArray.length + 1);
			newArray[newArray.length - 1] = 0.0;
			return newArray;
		}

		@Override
		public BoundaryCondition[] getBoundaryConditionsAtLowerBoundary(
				final FiniteDifferenceProduct product,
				double time,
				final double... stateVariables) {

			if(time == 0.0) {
				time = 1e-6;
			}

			final AsianOption option = (AsianOption) product;
			final double maturity = option.getMaturity();
			final double strike = option.getStrike();
			final CallOrPut callOrPut = option.getCallOrPut();
			final AsianStrike asianStrike = option.getAsianStrike();

			final double r = -Math.log(getRiskFreeCurve().getDiscountFactor(time)) / time;
			final double delta = maturity - time;
			final double discount = Math.exp(-r * delta);

			final BoundaryCondition[] result = new BoundaryCondition[2];

			final double I = stateVariables.length > 1 ? stateVariables[1] : 0.0;
			final double averageSoFar = I / maturity;

			/*
			 * Lower boundary in S: S = 0
			 */
			final double lowerSValue;

			if(asianStrike == AsianStrike.FIXED_STRIKE) {
				if(callOrPut == CallOrPut.CALL) {
					/*
					 * At S = 0 and assuming the process remains there,
					 * the final average is A(T) = I/T.
					 * Hence the fixed-strike call value is the discounted payoff
					 * max(A(T) - K, 0).
					 */
					lowerSValue = discount * Math.max(averageSoFar - strike, 0.0);
				}
				else {
					lowerSValue = discount * Math.max(strike - averageSoFar, 0.0);
				}
			}
			else if(asianStrike == AsianStrike.FLOATING_STRIKE) {
				if(callOrPut == CallOrPut.CALL) {
					/*
					 * payoff = max(S(T) - A(T), 0), and at S=0 this is 0
					 */
					lowerSValue = 0.0;
				}
				else {
					/*
					 * payoff = max(A(T) - S(T), 0), and at S=0 this becomes A(T).
					 * Since S=0 from now on, the future average contribution vanishes,
					 * so A(T) = I/T and its discounted value is discount * I/T.
					 */
					lowerSValue = discount * averageSoFar;
				}
			}
			else {
				throw new IllegalArgumentException("Unrecognized strike type.");
			}

			result[0] = StandardBoundaryCondition.dirichlet(lowerSValue);

			/*
			 * Lower boundary in I: I = 0
			 * Leave PDE row intact.
			 */
			result[1] = StandardBoundaryCondition.none();

			return result;
		}

		@Override
		public BoundaryCondition[] getBoundaryConditionsAtUpperBoundary(
				final FiniteDifferenceProduct product,
				double time,
				final double... stateVariables) {

			if(time == 0.0) {
				time = 1e-6;
			}

			final AsianOption option = (AsianOption) product;
			final double maturity = option.getMaturity();
			final double strike = option.getStrike();
			final CallOrPut callOrPut = option.getCallOrPut();
			final AsianStrike asianStrike = option.getAsianStrike();

			final double r = -Math.log(getRiskFreeCurve().getDiscountFactor(time)) / time;
			final double q = -Math.log(getDividendYieldCurve().getDiscountFactor(time)) / time;
			final double delta = maturity - time;
			final double discount = Math.exp(-r * delta);

			final double S = stateVariables.length > 0 ? stateVariables[0] : 0.0;
			final double I = stateVariables.length > 1 ? stateVariables[1] : 0.0;

			final BoundaryCondition[] result = new BoundaryCondition[2];

			final double averageSoFar = I / maturity;

			final double discountedExpectedRemainingAverageContribution;
			if(Math.abs(r - q) > 1E-12) {
				discountedExpectedRemainingAverageContribution =
						(S / maturity) * (Math.exp(-q * delta) - Math.exp(-r * delta)) / (r - q);
			}
			else {
				discountedExpectedRemainingAverageContribution =
						(S / maturity) * delta * Math.exp(-r * delta);
			}

			final double discountedExpectedAverage =
					discount * averageSoFar + discountedExpectedRemainingAverageContribution;

			final double discountedExpectedSpot =
					S * Math.exp(-q * delta);

			if(asianStrike == AsianStrike.FIXED_STRIKE) {

				/*
				 * Upper boundary in S: deep ITM fixed-strike call asymptotic
				 * V ~ E^disc[A(T)] - E^disc[K]
				 */
				final double upperSCallValue =
						discountedExpectedAverage - discount * strike;

				result[0] = StandardBoundaryCondition.dirichlet(
						(callOrPut == CallOrPut.CALL) ? upperSCallValue : 0.0);

				/*
				 * Upper boundary in I:
				 * do NOT impose a hard Dirichlet value at the artificial I_max boundary.
				 * Let the PDE determine that row.
				 */
				result[1] = StandardBoundaryCondition.none();
			}
			else if(asianStrike == AsianStrike.FLOATING_STRIKE) {

				/*
				 * Upper boundary in S: deep ITM floating-strike call asymptotic
				 * V ~ E^disc[S(T)] - E^disc[A(T)]
				 *
				 * Do not apply Math.max here. The asymptotic linear form is smoother
				 * and more appropriate as a far-field boundary condition.
				 */
				final double upperSCallValue =
						discountedExpectedSpot - discountedExpectedAverage;

				result[0] = StandardBoundaryCondition.dirichlet(
						(callOrPut == CallOrPut.CALL) ? upperSCallValue : 0.0);

				/*
				 * Upper boundary in I:
				 * avoid forcing a Dirichlet value at finite I_max.
				 */
				result[1] = StandardBoundaryCondition.none();
			}
			else {
				throw new IllegalArgumentException("Unrecognized strike type.");
			}

			return result;
		}
		@Override
		public FiniteDifferenceEquityModel getCloneWithModifiedSpaceTimeDiscretization(
				final SpaceTimeDiscretization newSpaceTimeDiscretization) {

			return new LiftedFDMBlackScholesModelDecorator(delegate, newSpaceTimeDiscretization);
		}
	}
}