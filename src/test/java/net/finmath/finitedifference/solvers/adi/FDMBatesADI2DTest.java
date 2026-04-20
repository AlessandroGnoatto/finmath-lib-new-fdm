package net.finmath.finitedifference.solvers.adi;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.exception.CalculationException;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBatesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Unit tests for {@link FDMBatesADI2D}.
 *
 * @author Alessandro Gnoatto
 */
public class FDMBatesADI2DTest {

	private static final double INITIAL_VALUE = 100.0;
	private static final double INITIAL_VARIANCE = 0.04;

	private static final double STRIKE = 100.0;
	private static final double MATURITY = 1.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.0;

	private static final double KAPPA = 1.5;
	private static final double THETA_V = 0.04;
	private static final double VOL_OF_VOL = 0.30;
	private static final double RHO = -0.7;

	private static final double JUMP_INTENSITY = 0.40;
	private static final double JUMP_MEAN = -0.10;
	private static final double JUMP_STD_DEV = 0.20;
	private static final double LOWER_INTEGRATION_BOUND = -2.0;
	private static final double UPPER_INTEGRATION_BOUND = 2.0;

	private static final double PDE_THETA = 1.0;
	private static final int NUMBER_OF_TIME_STEPS = 200;
	private static final int NUMBER_OF_SPOT_STEPS = 240;
	private static final int NUMBER_OF_VARIANCE_STEPS = 100;

	private static final int QUADRATURE_POINTS_PER_SIDE = 200;

	private static final double MONOTONICITY_TOLERANCE = 1E-6;
	private static final double ZERO_JUMP_TOLERANCE = 1E-8;
	private static final double FOURIER_TOLERANCE = 1.00;

	@Test
	public void testEuropeanCallPriceIsFiniteAndNonNegative() {

		final SpaceTimeDiscretization discretization = createDiscretization();

		final double price = getEuropeanCallPriceAtInitialNode(
				createBatesModel(discretization, JUMP_INTENSITY),
				QUADRATURE_POINTS_PER_SIDE
		);

		assertTrue(Double.isFinite(price));
		assertTrue(price >= -1E-10);
	}

	@Test
	public void testEuropeanCallSurfaceIsMonotoneInSpotForInitialVarianceSlice() {

		final SpaceTimeDiscretization discretization = createDiscretization();
		final double[] values = getEuropeanCallValuesAtTimeZero(
				createBatesModel(discretization, JUMP_INTENSITY),
				QUADRATURE_POINTS_PER_SIDE
		);

		final double[] spotGrid = discretization.getSpaceGrid(0).getGrid();
		final double[] varianceGrid = discretization.getSpaceGrid(1).getGrid();

		final int initialVarianceIndex = getGridIndex(varianceGrid, INITIAL_VARIANCE);
		final int n0 = spotGrid.length;

		for(int i = 0; i < n0 - 1; i++) {
			final double leftValue = values[flatten(i, initialVarianceIndex, n0)];
			final double rightValue = values[flatten(i + 1, initialVarianceIndex, n0)];

			assertTrue(
					"European call values should be non-decreasing in spot on the initial-variance slice. "
							+ "Violation at spot indices " + i + " and " + (i + 1)
							+ ": " + leftValue + " > " + rightValue,
					leftValue <= rightValue + MONOTONICITY_TOLERANCE
			);
		}
	}

	@Test
	public void testZeroJumpBatesAgreesWithHeston() {

		final SpaceTimeDiscretization discretization = createDiscretization();

		final FDMBatesModel zeroJumpBatesModel = createBatesModel(discretization, 0.0);
		final FDMHestonModel hestonModel = createHestonModel(discretization);

		final double batesPrice = getEuropeanCallPriceAtInitialNode(
				zeroJumpBatesModel,
				QUADRATURE_POINTS_PER_SIDE
		);

		final double hestonPrice = getEuropeanCallPriceAtInitialNode(hestonModel);

		assertEquals(hestonPrice, batesPrice, ZERO_JUMP_TOLERANCE);
	}

	@Test
	public void testEuropeanCallAgainstFourierBenchmark() throws CalculationException {

		final SpaceTimeDiscretization discretization = createDiscretization();

		final double finiteDifferencePrice = getEuropeanCallPriceAtInitialNode(
				createBatesModel(discretization, JUMP_INTENSITY),
				QUADRATURE_POINTS_PER_SIDE
		);

		final double fourierPrice = getFourierEuropeanCallValue();

		assertTrue(finiteDifferencePrice >= -1E-10);
		assertTrue(fourierPrice >= -1E-10);

		assertEquals(fourierPrice, finiteDifferencePrice, FOURIER_TOLERANCE);
	}

	private FDMBatesModel createBatesModel(
			final SpaceTimeDiscretization discretization,
			final double jumpIntensity) {

		return new FDMBatesModel(
				INITIAL_VALUE,
				INITIAL_VARIANCE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				KAPPA,
				THETA_V,
				VOL_OF_VOL,
				RHO,
				jumpIntensity,
				JUMP_MEAN,
				JUMP_STD_DEV,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				discretization
		);
	}

	private FDMHestonModel createHestonModel(final SpaceTimeDiscretization discretization) {
		return new FDMHestonModel(
				INITIAL_VALUE,
				INITIAL_VARIANCE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				KAPPA,
				THETA_V,
				VOL_OF_VOL,
				RHO,
				discretization
		);
	}

	private double getEuropeanCallPriceAtInitialNode(
			final FDMBatesModel model,
			final int quadraturePointsPerSide) {

		final double[] values = getEuropeanCallValuesAtTimeZero(model, quadraturePointsPerSide);

		final double[] spotGrid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double[] varianceGrid = model.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid();

		final int initialSpotIndex = getGridIndex(spotGrid, INITIAL_VALUE);
		final int initialVarianceIndex = getGridIndex(varianceGrid, INITIAL_VARIANCE);

		return values[flatten(initialSpotIndex, initialVarianceIndex, spotGrid.length)];
	}

	private double[] getEuropeanCallValuesAtTimeZero(
			final FDMBatesModel model,
			final int quadraturePointsPerSide) {

		final EuropeanOption option = new EuropeanOption(MATURITY, STRIKE, CallOrPut.CALL);

		final FDMBatesADI2D solver = new FDMBatesADI2D(
				model,
				option,
				model.getSpaceTimeDiscretization(),
				new EuropeanExercise(MATURITY),
				quadraturePointsPerSide
		);

		return solver.getValue(
				0.0,
				MATURITY,
				(stock, variance) -> Math.max(stock - STRIKE, 0.0)
		);
	}

	private double getEuropeanCallPriceAtInitialNode(final FDMHestonModel model) {

		final EuropeanOption option = new EuropeanOption(MATURITY, STRIKE, CallOrPut.CALL);

		final FDMHestonADI2D solver = new FDMHestonADI2D(
				model,
				option,
				model.getSpaceTimeDiscretization(),
				new EuropeanExercise(MATURITY)
		);

		final double[] values = solver.getValue(
				0.0,
				MATURITY,
				(stock, variance) -> Math.max(stock - STRIKE, 0.0)
		);

		final double[] spotGrid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double[] varianceGrid = model.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid();

		final int initialSpotIndex = getGridIndex(spotGrid, INITIAL_VALUE);
		final int initialVarianceIndex = getGridIndex(varianceGrid, INITIAL_VARIANCE);

		return values[flatten(initialSpotIndex, initialVarianceIndex, spotGrid.length)];
	}

	private double getFourierEuropeanCallValue() throws CalculationException {

			final double fourierInitialVariance = INITIAL_VARIANCE;
			final double alpha = KAPPA * THETA_V;
			final double beta = KAPPA;
			final double k = Math.exp(JUMP_MEAN + 0.5 * JUMP_STD_DEV * JUMP_STD_DEV) - 1.0;
			final double delta = JUMP_STD_DEV;

			final net.finmath.fouriermethod.models.CharacteristicFunctionModel characteristicFunctionBates =
					new net.finmath.fouriermethod.models.BatesModel(
							INITIAL_VALUE,
							RISK_FREE_RATE,
							fourierInitialVariance,
							alpha,
							beta,
							VOL_OF_VOL,
							RHO,
							JUMP_INTENSITY,
							0.0,
							k,
							delta
					);

			final net.finmath.fouriermethod.products.FourierTransformProduct europeanFourier =
					new net.finmath.fouriermethod.products.EuropeanOption(
							MATURITY,
							STRIKE
					);
		return europeanFourier.getValue(characteristicFunctionBates);
	}

	private SpaceTimeDiscretization createDiscretization() {

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		/*
		 * Spot grid: [40, 160] with 240 steps gives deltaS = 0.5 and S0 = 100 exactly on-grid.
		 * Variance grid: [0.0, 0.2] with 100 steps gives deltav = 0.002 and v0 = 0.04 exactly on-grid.
		 */
		final Grid spotGrid = new UniformGrid(NUMBER_OF_SPOT_STEPS, 40.0, 160.0);
		final Grid varianceGrid = new UniformGrid(NUMBER_OF_VARIANCE_STEPS, 0.0, 0.2);

		return new SpaceTimeDiscretization(
				new Grid[] { spotGrid, varianceGrid },
				timeDiscretization,
				PDE_THETA,
				new double[] { INITIAL_VALUE, INITIAL_VARIANCE }
		);
	}

	private int flatten(final int i0, final int i1, final int n0) {
		return i0 + i1 * n0;
	}

	private int getGridIndex(final double[] grid, final double x) {
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - x) < 1E-12) {
				return i;
			}
		}
		throw new IllegalArgumentException("Point is not a grid node.");
	}
}