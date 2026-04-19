package net.finmath.finitedifference.solvers;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMMertonModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.finitedifference.solvers.adi.BarrierPDEMode;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for {@link FDMSolverFactory}.
 *
 * @author Alessandro Gnoatto
 */
public class FDMSolverFactoryTest {

	private static final double INITIAL_VALUE = 100.0;
	private static final double STRIKE = 100.0;
	private static final double MATURITY = 1.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.02;
	private static final double VOLATILITY = 0.20;

	private static final double JUMP_INTENSITY = 0.40;
	private static final double JUMP_MEAN = -0.10;
	private static final double JUMP_STD_DEV = 0.25;
	private static final double LOWER_INTEGRATION_BOUND = -2.0;
	private static final double UPPER_INTEGRATION_BOUND = 2.0;

	private static final double THETA = 1.0;
	private static final int NUMBER_OF_TIME_STEPS = 20;
	private static final int NUMBER_OF_SPACE_STEPS = 80;

	@Test
	public void testBlackScholesModelIsRoutedToPlainThetaSolver() {

		final SpaceTimeDiscretization discretization = createDiscretization();
		final EuropeanOption option = new EuropeanOption(MATURITY, STRIKE, CallOrPut.CALL);
		final Exercise exercise = new EuropeanExercise(MATURITY);

		final FDMBlackScholesModel model = new FDMBlackScholesModel(
				INITIAL_VALUE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				discretization
		);

		final FDMSolver solver = FDMSolverFactory.createSolver(model, option, exercise);

		assertTrue(solver instanceof FDMThetaMethod1D);
	}

	@Test
	public void testMertonModelIsRoutedToJumpThetaSolver() {

		final SpaceTimeDiscretization discretization = createDiscretization();
		final EuropeanOption option = new EuropeanOption(MATURITY, STRIKE, CallOrPut.CALL);
		final Exercise exercise = new EuropeanExercise(MATURITY);

		final FDMMertonModel model = new FDMMertonModel(
				INITIAL_VALUE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				JUMP_INTENSITY,
				JUMP_MEAN,
				JUMP_STD_DEV,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				discretization
		);

		final FDMSolver solver = FDMSolverFactory.createSolver(model, option, exercise);

		assertTrue(solver instanceof FDMThetaMethod1DJump);
	}

	@Test
	public void testBarrierAwareOverloadStillRoutesOneDimensionalMertonToJumpThetaSolver() {

		final SpaceTimeDiscretization discretization = createDiscretization();
		final EuropeanOption option = new EuropeanOption(MATURITY, STRIKE, CallOrPut.CALL);
		final Exercise exercise = new EuropeanExercise(MATURITY);

		final FDMMertonModel model = new FDMMertonModel(
				INITIAL_VALUE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				JUMP_INTENSITY,
				JUMP_MEAN,
				JUMP_STD_DEV,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				discretization
		);

		final FDMSolver solver = FDMSolverFactory.createSolver(
				model,
				option,
				exercise,
				BarrierPDEMode.OUT_STANDARD,
				null
		);

		assertTrue(solver instanceof FDMThetaMethod1DJump);
	}

	private SpaceTimeDiscretization createDiscretization() {

		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				NUMBER_OF_TIME_STEPS,
				MATURITY / NUMBER_OF_TIME_STEPS
		);

		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS, 40.0, 160.0);

		return new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { INITIAL_VALUE }
		);
	}
}