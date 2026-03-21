package net.finmath.finitedifference.assetderivativevaluation.products;

import net.finmath.finitedifference.FiniteDifferenceExerciseUtil;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBachelierModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMCevModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMThetaMethod1D;
import net.finmath.finitedifference.solvers.FDMThetaMethod2D;
import net.finmath.finitedifference.solvers.adi.FDMHestonADI2D;
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.CallOrPut;

/**
 * Finite difference valuation of a Bermudan option on a single asset.
 *
 * <p>
 * Exercise times are specified in running time and converted internally to the
 * solver's time-to-maturity coordinates through
 * {@link FiniteDifferenceExerciseUtil}.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class BermudanOption implements FiniteDifferenceProduct {

	private final String underlyingName;
	private final double maturity;
	private final double strike;
	private final CallOrPut callOrPutSign;
	private final Exercise exercise;

	public BermudanOption(
			final String underlyingName,
			final double[] exerciseTimes,
			final double strike,
			final double callOrPutSign) {
		this(
				underlyingName,
				exerciseTimes,
				strike,
				mapCallOrPut(callOrPutSign)
		);
	}

	public BermudanOption(
			final String underlyingName,
			final double[] exerciseTimes,
			final double strike,
			final CallOrPut callOrPutSign) {

		if(exerciseTimes == null || exerciseTimes.length == 0) {
			throw new IllegalArgumentException("Exercise times must not be null or empty.");
		}

		this.underlyingName = underlyingName;
		this.exercise = new BermudanExercise(exerciseTimes);
		this.maturity = this.exercise.getMaturity();
		this.strike = strike;
		this.callOrPutSign = callOrPutSign;
	}

	public BermudanOption(
			final double[] exerciseTimes,
			final double strike,
			final double callOrPutSign) {
		this(null, exerciseTimes, strike, callOrPutSign);
	}

	public BermudanOption(
			final double[] exerciseTimes,
			final double strike,
			final CallOrPut callOrPutSign) {
		this(null, exerciseTimes, strike, callOrPutSign);
	}

	@Override
	public double[] getValue(final double evaluationTime, final FiniteDifferenceEquityModel model) {
		final SpaceTimeDiscretization refinedDiscretization = getRefinedSpaceTimeDiscretization(model);
		final FDMSolver solver = createSolver(model, refinedDiscretization);

		if(callOrPutSign == CallOrPut.CALL) {
			return solver.getValue(
					evaluationTime,
					maturity,
					assetValue -> Math.max(assetValue - strike, 0.0)
			);
		}
		else {
			return solver.getValue(
					evaluationTime,
					maturity,
					assetValue -> Math.max(strike - assetValue, 0.0)
			);
		}
	}

	@Override
	public double[][] getValues(final FiniteDifferenceEquityModel model) {
		final SpaceTimeDiscretization refinedDiscretization = getRefinedSpaceTimeDiscretization(model);
		final FDMSolver solver = createSolver(model, refinedDiscretization);

		if(callOrPutSign == CallOrPut.CALL) {
			return solver.getValues(maturity, assetValue -> Math.max(assetValue - strike, 0.0));
		}
		else {
			return solver.getValues(maturity, assetValue -> Math.max(strike - assetValue, 0.0));
		}
	}

	private FDMSolver createSolver(
			final FiniteDifferenceEquityModel model,
			final SpaceTimeDiscretization spaceTimeDiscretization) {

		if(model instanceof FDMBlackScholesModel) {
			return new FDMThetaMethod1D(model, this, spaceTimeDiscretization, exercise);
		}
		else if(model instanceof FDMCevModel) {
			return new FDMThetaMethod1D(model, this, spaceTimeDiscretization, exercise);
		}
		else if(model instanceof FDMBachelierModel) {
			return new FDMThetaMethod1D(model, this, spaceTimeDiscretization, exercise);
		}
		else if(model instanceof FDMHestonModel) {
			return new FDMHestonADI2D((FDMHestonModel) model, this, model.getSpaceTimeDiscretization(), exercise);//FDMThetaMethod2D(model, this, spaceTimeDiscretization, exercise);
		}
		else {
			throw new IllegalArgumentException(
					"Unsupported model type: " + model.getClass().getName());
		}
	}

	private SpaceTimeDiscretization getRefinedSpaceTimeDiscretization(final FiniteDifferenceEquityModel model) {
		final SpaceTimeDiscretization base = model.getSpaceTimeDiscretization();

		final var refinedTimeDiscretization =
				FiniteDifferenceExerciseUtil.refineTimeDiscretization(
						base.getTimeDiscretization(),
						exercise
				);

		if(base.getNumberOfSpaceGrids() == 1) {
			return new SpaceTimeDiscretization(
					base.getSpaceGrid(0),
					refinedTimeDiscretization,
					base.getTheta(),
					new double[] { base.getCenter(0) }
			);
		}

		final int numberOfSpaceGrids = base.getNumberOfSpaceGrids();
		final Grid[] spaceGrids = new Grid[numberOfSpaceGrids];
		final double[] center = new double[numberOfSpaceGrids];

		for(int i = 0; i < numberOfSpaceGrids; i++) {
			spaceGrids[i] = base.getSpaceGrid(i);
			center[i] = base.getCenter(i);
		}

		return new SpaceTimeDiscretization(
				spaceGrids,
				refinedTimeDiscretization,
				base.getTheta(),
				center
		);
	}

	private static CallOrPut mapCallOrPut(final double callOrPutSign) {
		if(callOrPutSign == 1.0) {
			return CallOrPut.CALL;
		}
		if(callOrPutSign == -1.0) {
			return CallOrPut.PUT;
		}
		throw new IllegalArgumentException("Unknown option type");
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

	public Exercise getExercise() {
		return exercise;
	}
}