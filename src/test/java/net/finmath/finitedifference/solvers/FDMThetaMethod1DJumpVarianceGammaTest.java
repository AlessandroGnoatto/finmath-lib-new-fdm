package net.finmath.finitedifference.solvers;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMVarianceGammaModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Unit tests for {@link FDMThetaMethod1DJump} under the
 * {@link FDMVarianceGammaModel}.
 *
 * @author Alessandro Gnoatto
 */
public class FDMThetaMethod1DJumpVarianceGammaTest {

	private static final double INITIAL_VALUE = 100.0;
	private static final double STRIKE = 100.0;
	private static final double MATURITY = 1.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.02;

	private static final double SIGMA = 0.30;
	private static final double NU = 0.20;
	private static final double THETA_PARAMETER = -0.10;

	private static final double LOWER_INTEGRATION_BOUND = -2.0;
	private static final double UPPER_INTEGRATION_BOUND = 2.0;

	private static final double PDE_THETA = 1.0;
	private static final int NUMBER_OF_TIME_STEPS = 300;
	private static final int NUMBER_OF_SPACE_STEPS = 240;

	private static final double MONOTONICITY_TOLERANCE = 1E-6;
	private static final double QUADRATURE_REFINEMENT_TOLERANCE = 0.15;

	@Test
	public void testEuropeanCallPriceIsFiniteAndNonNegative() {

		final SpaceTimeDiscretization discretization = createDiscretization();
		final double[] sNodes = discretization.getSpaceGrid(0).getGrid();

		assertTrue("Spot must be a grid node for this test.", isGridNode(sNodes, INITIAL_VALUE));

		final double callPrice = getEuropeanOptionPriceAtSpot(
				createModel(discretization, THETA_PARAMETER),
				CallOrPut.CALL,
				100
		);

		assertTrue(Double.isFinite(callPrice));
		assertTrue(callPrice >= -1E-10);
	}

	@Test
	public void testEuropeanPutPriceIsFiniteAndNonNegative() {

		final SpaceTimeDiscretization discretization = createDiscretization();
		final double[] sNodes = discretization.getSpaceGrid(0).getGrid();

		assertTrue("Spot must be a grid node for this test.", isGridNode(sNodes, INITIAL_VALUE));

		final double putPrice = getEuropeanOptionPriceAtSpot(
				createModel(discretization, THETA_PARAMETER),
				CallOrPut.PUT,
				100
		);

		assertTrue(Double.isFinite(putPrice));
		assertTrue(putPrice >= -1E-10);
	}

	@Test
	public void testEuropeanCallSurfaceIsMonotoneInSpot() {

		final SpaceTimeDiscretization discretization = createDiscretization();

		final double[] values = getEuropeanOptionValuesAtTimeZero(
				createModel(discretization, THETA_PARAMETER),
				CallOrPut.CALL,
				100
		);

		for(int i = 0; i < values.length - 1; i++) {
			assertTrue(
					"European call values should be non-decreasing in spot. Violation at indices "
							+ i + " and " + (i + 1) + ": " + values[i] + " > " + values[i + 1],
					values[i] <= values[i + 1] + MONOTONICITY_TOLERANCE
			);
		}
	}

	@Test
	public void testEuropeanPutSurfaceIsMonotoneInSpot() {

		final SpaceTimeDiscretization discretization = createDiscretization();

		final double[] values = getEuropeanOptionValuesAtTimeZero(
				createModel(discretization, THETA_PARAMETER),
				CallOrPut.PUT,
				100
		);

		for(int i = 0; i < values.length - 1; i++) {
			assertTrue(
					"European put values should be non-increasing in spot. Violation at indices "
							+ i + " and " + (i + 1) + ": " + values[i] + " < " + values[i + 1],
					values[i] + MONOTONICITY_TOLERANCE >= values[i + 1]
			);
		}
	}

	@Test
	public void testQuadratureRefinementStabilityForEuropeanCall() {

		final SpaceTimeDiscretization discretization = createDiscretization();
		final FDMVarianceGammaModel model = createModel(discretization, THETA_PARAMETER);

		final double priceWith50 = getEuropeanOptionPriceAtSpot(model, CallOrPut.CALL, 50);
		final double priceWith100 = getEuropeanOptionPriceAtSpot(model, CallOrPut.CALL, 100);
		final double priceWith200 = getEuropeanOptionPriceAtSpot(model, CallOrPut.CALL, 200);

		assertTrue(Math.abs(priceWith100 - priceWith50) < QUADRATURE_REFINEMENT_TOLERANCE);
		assertTrue(Math.abs(priceWith200 - priceWith100) < QUADRATURE_REFINEMENT_TOLERANCE);
	}

	@Test
	public void testSymmetricVarianceGammaCallPriceIsFiniteAndNonNegative() {

		final SpaceTimeDiscretization discretization = createDiscretization();
		final double symmetricTheta = 0.0;

		final double callPrice = getEuropeanOptionPriceAtSpot(
				createModel(discretization, symmetricTheta),
				CallOrPut.CALL,
				100
		);

		assertTrue(Double.isFinite(callPrice));
		assertTrue(callPrice >= -1E-10);
	}

	private FDMVarianceGammaModel createModel(
			final SpaceTimeDiscretization discretization,
			final double thetaParameter) {

		return new FDMVarianceGammaModel(
				INITIAL_VALUE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				SIGMA,
				NU,
				thetaParameter,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				discretization
		);
	}

	private double getEuropeanOptionPriceAtSpot(
			final FDMVarianceGammaModel model,
			final CallOrPut callOrPut,
			final int quadraturePointsPerSide) {

		final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double[] values = getEuropeanOptionValuesAtTimeZero(model, callOrPut, quadraturePointsPerSide);

		return values[getGridIndex(sNodes, INITIAL_VALUE)];
	}

	private double[] getEuropeanOptionValuesAtTimeZero(
			final FDMVarianceGammaModel model,
			final CallOrPut callOrPut,
			final int quadraturePointsPerSide) {

		final EuropeanOption option = new EuropeanOption(MATURITY, STRIKE, callOrPut);

		final FDMThetaMethod1DJump solver = new FDMThetaMethod1DJump(
				model,
				option,
				model.getSpaceTimeDiscretization(),
				new EuropeanExercise(MATURITY),
				quadraturePointsPerSide
		);

		if(callOrPut == CallOrPut.CALL) {
			return solver.getValue(
					0.0,
					MATURITY,
					stock -> Math.max(stock - STRIKE, 0.0)
			);
		}
		else {
			return solver.getValue(
					0.0,
					MATURITY,
					stock -> Math.max(STRIKE - stock, 0.0)
			);
		}
	}

	private SpaceTimeDiscretization createDiscretization() {

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		/*
		 * [40,160] with 240 steps gives deltaS = 0.5 and S0 = 100 exactly on-grid.
		 */
		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS, 40.0, 160.0);

		return new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				PDE_THETA,
				new double[] { INITIAL_VALUE }
		);
	}

	private boolean isGridNode(final double[] grid, final double x) {
		for(final double node : grid) {
			if(Math.abs(node - x) < 1E-12) {
				return true;
			}
		}
		return false;
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