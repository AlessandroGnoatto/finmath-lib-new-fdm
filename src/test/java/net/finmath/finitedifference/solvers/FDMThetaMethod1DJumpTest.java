package net.finmath.finitedifference.solvers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMMertonModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Unit tests for {@link FDMThetaMethod1DJump}.
 *
 * <p>
 * These tests assume that the boundary factory can resolve
 * {@code EuropeanOptionMertonModelBoundary} for the plain {@link FDMMertonModel}.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class FDMThetaMethod1DJumpTest {

	private static final double SPOT = 100.0;
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
	private static final int NUMBER_OF_TIME_STEPS = 300;
	private static final int NUMBER_OF_SPACE_STEPS = 240;
	private static final int QUADRATURE_POINTS_PER_SIDE = 200;

	private static final double ZERO_INTENSITY_TOLERANCE = 1E-8;
	private static final double MONOTONICITY_TOLERANCE = 1E-8;

	@Test
	public void testZeroIntensityAgreesWithPlainThetaMethodForEuropeanCall() {

		final SpaceTimeDiscretization discretization = createDiscretization();
		final double[] sNodes = discretization.getSpaceGrid(0).getGrid();

		assertTrue("Spot must be a grid node for this regression test.", isGridNode(sNodes, SPOT));

		final EuropeanOption option = new EuropeanOption(MATURITY, STRIKE, CallOrPut.CALL);

		final FDMBlackScholesModel blackScholesModel = new FDMBlackScholesModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				discretization
		);

		final FDMMertonModel zeroJumpMertonModel = new FDMMertonModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				0.0,
				JUMP_MEAN,
				JUMP_STD_DEV,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				discretization
		);

		final FDMThetaMethod1D plainSolver = new FDMThetaMethod1D(
				blackScholesModel,
				option,
				discretization,
				new EuropeanExercise(MATURITY)
		);

		final FDMThetaMethod1DJump jumpSolver = new FDMThetaMethod1DJump(
				zeroJumpMertonModel,
				option,
				discretization,
				new EuropeanExercise(MATURITY),
				QUADRATURE_POINTS_PER_SIDE
		);

		final double[] plainValues = plainSolver.getValue(
				0.0,
				MATURITY,
				stock -> Math.max(stock - STRIKE, 0.0)
		);

		final double[] jumpValues = jumpSolver.getValue(
				0.0,
				MATURITY,
				stock -> Math.max(stock - STRIKE, 0.0)
		);

		final double plainPrice = plainValues[getGridIndex(sNodes, SPOT)];
		final double jumpPrice = jumpValues[getGridIndex(sNodes, SPOT)];

		assertEquals(plainPrice, jumpPrice, ZERO_INTENSITY_TOLERANCE);
	}

	@Test
	public void testEuropeanCallWithJumpsIsNonNegativeAndBelowDiscountedSpot() {

		final SpaceTimeDiscretization discretization = createDiscretization();
		final double[] sNodes = discretization.getSpaceGrid(0).getGrid();

		final EuropeanOption option = new EuropeanOption(MATURITY, STRIKE, CallOrPut.CALL);

		final FDMMertonModel mertonModel = new FDMMertonModel(
				SPOT,
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

		final FDMThetaMethod1DJump jumpSolver = new FDMThetaMethod1DJump(
				mertonModel,
				option,
				discretization,
				new EuropeanExercise(MATURITY),
				QUADRATURE_POINTS_PER_SIDE
		);

		final double[] values = jumpSolver.getValue(
				0.0,
				MATURITY,
				stock -> Math.max(stock - STRIKE, 0.0)
		);

		final double priceAtSpot = interpolateAtSpot(values, sNodes, SPOT);
		final double discountedSpotUpperBound =
				SPOT * mertonModel.getDividendYieldCurve().getDiscountFactor(MATURITY);

		assertTrue("Jump-diffusion call value must be non-negative.", priceAtSpot >= -1E-10);
		assertTrue(
				"Jump-diffusion call value must not exceed discounted spot upper bound.",
				priceAtSpot <= discountedSpotUpperBound + 1E-6
		);
	}

	@Test
	public void testEuropeanCallSurfaceIsMonotoneInSpot() {

		final SpaceTimeDiscretization discretization = createDiscretization();

		final EuropeanOption option = new EuropeanOption(MATURITY, STRIKE, CallOrPut.CALL);

		final FDMMertonModel mertonModel = new FDMMertonModel(
				SPOT,
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

		final FDMThetaMethod1DJump jumpSolver = new FDMThetaMethod1DJump(
				mertonModel,
				option,
				discretization,
				new EuropeanExercise(MATURITY),
				QUADRATURE_POINTS_PER_SIDE
		);

		final double[] values = jumpSolver.getValue(
				0.0,
				MATURITY,
				stock -> Math.max(stock - STRIKE, 0.0)
		);

		for(int i = 0; i < values.length - 1; i++) {
			assertTrue(
					"European call values should be non-decreasing in spot. "
							+ "Violation at indices " + i + " and " + (i + 1)
							+ ": " + values[i] + " > " + values[i + 1],
					values[i] <= values[i + 1] + MONOTONICITY_TOLERANCE
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
		 * Use a uniform grid where spot is exactly a node:
		 * [40, 160] with 240 steps gives deltaS = 0.5 and S0 = 100 exactly on-grid.
		 */
		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS, 40.0, 160.0);

		return new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { SPOT }
		);
	}

	private double interpolateAtSpot(
			final double[] values,
			final double[] sNodes,
			final double spot) {

		assertTrue(
				"Spot must lie inside the grid domain.",
				spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12
		);

		if(isGridNode(sNodes, spot)) {
			return values[getGridIndex(sNodes, spot)];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(sNodes, values);

		return interpolation.value(spot);
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