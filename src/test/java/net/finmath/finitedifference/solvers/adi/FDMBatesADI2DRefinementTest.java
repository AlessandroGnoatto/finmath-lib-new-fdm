package net.finmath.finitedifference.solvers.adi;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBatesModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Refinement tests for {@link FDMBatesADI2D}.
 *
 * <p>
 * These tests are intentionally lighter than the full Bates benchmark tests,
 * since the two-dimensional jump solver is computationally expensive. The goal
 * is not to prove asymptotic convergence rigorously, but to verify that the
 * European call price is reasonably stable under:
 * </p>
 * <ul>
 *   <li>refinement of the jump quadrature,</li>
 *   <li>refinement of the space-time grid.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class FDMBatesADI2DRefinementTest {

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

	/*
	 * Tolerances are deliberately moderate: Bates in 2D with an explicit jump
	 * term is already numerically demanding, and these are regression / stability
	 * tests, not asymptotic error proofs.
	 */
	private static final double QUADRATURE_REFINEMENT_TOLERANCE = 0.35;
	private static final double GRID_REFINEMENT_TOLERANCE = 0.60;

	@Test
	public void testQuadratureRefinementStability() {

		final SpaceTimeDiscretization discretization = createDiscretization(80, 120, 50);
		final FDMBatesModel model = createBatesModel(discretization);

		final double priceWith50 = getEuropeanCallPriceAtInitialNode(model, 50);
		final double priceWith100 = getEuropeanCallPriceAtInitialNode(model, 100);
		final double priceWith150 = getEuropeanCallPriceAtInitialNode(model, 150);

		assertTrue(Double.isFinite(priceWith50));
		assertTrue(Double.isFinite(priceWith100));
		assertTrue(Double.isFinite(priceWith150));

		assertTrue(priceWith50 >= -1E-10);
		assertTrue(priceWith100 >= -1E-10);
		assertTrue(priceWith150 >= -1E-10);

		assertTrue(Math.abs(priceWith100 - priceWith50) < QUADRATURE_REFINEMENT_TOLERANCE);
		assertTrue(Math.abs(priceWith150 - priceWith100) < QUADRATURE_REFINEMENT_TOLERANCE);
	}

	@Test
	public void testGridRefinementStability() {

		final double coarsePrice = getEuropeanCallPriceAtInitialNode(
				createBatesModel(createDiscretization(60, 100, 40)),
				100
		);

		final double mediumPrice = getEuropeanCallPriceAtInitialNode(
				createBatesModel(createDiscretization(80, 120, 50)),
				100
		);

		final double finePrice = getEuropeanCallPriceAtInitialNode(
				createBatesModel(createDiscretization(100, 140, 60)),
				100
		);

		assertTrue(Double.isFinite(coarsePrice));
		assertTrue(Double.isFinite(mediumPrice));
		assertTrue(Double.isFinite(finePrice));

		assertTrue(coarsePrice >= -1E-10);
		assertTrue(mediumPrice >= -1E-10);
		assertTrue(finePrice >= -1E-10);

		assertTrue(Math.abs(mediumPrice - coarsePrice) < GRID_REFINEMENT_TOLERANCE);
		assertTrue(Math.abs(finePrice - mediumPrice) < GRID_REFINEMENT_TOLERANCE);
	}

	private FDMBatesModel createBatesModel(final SpaceTimeDiscretization discretization) {
		return new FDMBatesModel(
				INITIAL_VALUE,
				INITIAL_VARIANCE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				KAPPA,
				THETA_V,
				VOL_OF_VOL,
				RHO,
				JUMP_INTENSITY,
				JUMP_MEAN,
				JUMP_STD_DEV,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				discretization
		);
	}

	private double getEuropeanCallPriceAtInitialNode(
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

	private SpaceTimeDiscretization createDiscretization(
			final int numberOfTimeSteps,
			final int numberOfSpotSteps,
			final int numberOfVarianceSteps) {

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						numberOfTimeSteps,
						MATURITY / numberOfTimeSteps
				);

		/*
		 * Spot grid: [40,160] with even step counts so that S0 = 100 lies exactly on-grid.
		 * Variance grid: [0.0,0.2] with multiples of 5 so that v0 = 0.04 lies exactly on-grid.
		 */
		final Grid spotGrid = new UniformGrid(numberOfSpotSteps, 40.0, 160.0);
		final Grid varianceGrid = new UniformGrid(numberOfVarianceSteps, 0.0, 0.2);

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