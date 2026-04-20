package net.finmath.finitedifference.solvers;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBatesModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.finitedifference.solvers.adi.FDMBatesADI2D;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for Bates routing in {@link FDMSolverFactory}.
 *
 * @author Alessandro Gnoatto
 */
public class FDMSolverFactoryBatesTest {

	private static final double INITIAL_VALUE = 100.0;
	private static final double INITIAL_VARIANCE = 0.04;
	private static final double STRIKE = 100.0;
	private static final double MATURITY = 1.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.02;

	private static final double KAPPA = 1.5;
	private static final double THETA_V = 0.04;
	private static final double SIGMA = 0.30;
	private static final double RHO = -0.7;

	private static final double JUMP_INTENSITY = 0.40;
	private static final double JUMP_MEAN = -0.10;
	private static final double JUMP_STD_DEV = 0.20;
	private static final double LOWER_INTEGRATION_BOUND = -2.0;
	private static final double UPPER_INTEGRATION_BOUND = 2.0;

	private static final double PDE_THETA = 1.0;
	private static final int NUMBER_OF_TIME_STEPS = 20;
	private static final int NUMBER_OF_SPOT_STEPS = 80;
	private static final int NUMBER_OF_VARIANCE_STEPS = 40;

	@Test
	public void testBatesModelIsRoutedToBatesADISolver() {

		final SpaceTimeDiscretization discretization = createDiscretization();
		final EuropeanOption option = new EuropeanOption(MATURITY, STRIKE, CallOrPut.CALL);
		final Exercise exercise = new EuropeanExercise(MATURITY);

		final FDMBatesModel model = new FDMBatesModel(
				INITIAL_VALUE,
				INITIAL_VARIANCE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				KAPPA,
				THETA_V,
				SIGMA,
				RHO,
				JUMP_INTENSITY,
				JUMP_MEAN,
				JUMP_STD_DEV,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				discretization
		);

		final FDMSolver solver = FDMSolverFactory.createSolver(model, option, exercise);

		assertTrue(solver instanceof FDMBatesADI2D);
	}

	private SpaceTimeDiscretization createDiscretization() {

		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				NUMBER_OF_TIME_STEPS,
				MATURITY / NUMBER_OF_TIME_STEPS
		);

		final Grid spotGrid = new UniformGrid(NUMBER_OF_SPOT_STEPS, 40.0, 160.0);
		final Grid varianceGrid = new UniformGrid(NUMBER_OF_VARIANCE_STEPS, 0.0, 0.25);

		return new SpaceTimeDiscretization(
				new Grid[] { spotGrid, varianceGrid },
				timeDiscretization,
				PDE_THETA,
				new double[] { INITIAL_VALUE, INITIAL_VARIANCE }
		);
	}
}