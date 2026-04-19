package net.finmath.finitedifference.assetderivativevaluation.models;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.MertonJumpComponent;

/**
 * Unit tests for {@link MertonJumpComponent}.
 *
 * <p>
 * These tests validate:
 * </p>
 * <ul>
 *   <li>basic metadata and constructor wiring,</li>
 *   <li>non-negativity of the Levy density,</li>
 *   <li>Gaussian symmetry around the jump mean,</li>
 *   <li>numerical recovery of the jump intensity by integrating the Levy density.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class MertonJumpComponentTest {

	private static final double TIME = 0.5;

	private static final double INTENSITY = 1.7;
	private static final double JUMP_MEAN = -0.2;
	private static final double JUMP_STD_DEV = 0.35;

	private static final double LOWER_BOUND = JUMP_MEAN - 8.0 * JUMP_STD_DEV;
	private static final double UPPER_BOUND = JUMP_MEAN + 8.0 * JUMP_STD_DEV;

	private static final double TOL = 1E-12;

	@Test
	public void testDefaultConstructorMetadata() {
		final MertonJumpComponent jumpComponent = new MertonJumpComponent(
				LOWER_BOUND,
				UPPER_BOUND,
				INTENSITY,
				JUMP_MEAN,
				JUMP_STD_DEV
		);

		assertEquals(0, jumpComponent.getStateVariableIndex());
		assertTrue(!jumpComponent.isStateDependent());
		assertTrue(jumpComponent.isFiniteActivity());
		assertTrue(jumpComponent.isFiniteVariation());

		assertEquals(LOWER_BOUND, jumpComponent.getLowerIntegrationBound(TIME, 100.0), TOL);
		assertEquals(UPPER_BOUND, jumpComponent.getUpperIntegrationBound(TIME, 100.0), TOL);

		assertEquals(INTENSITY, jumpComponent.getIntensity(), TOL);
		assertEquals(JUMP_MEAN, jumpComponent.getJumpMean(), TOL);
		assertEquals(JUMP_STD_DEV, jumpComponent.getJumpStdDev(), TOL);
	}

	@Test
	public void testCustomStateVariableIndex() {
		final int stateVariableIndex = 2;

		final MertonJumpComponent jumpComponent = new MertonJumpComponent(
				stateVariableIndex,
				LOWER_BOUND,
				UPPER_BOUND,
				INTENSITY,
				JUMP_MEAN,
				JUMP_STD_DEV
		);

		assertEquals(stateVariableIndex, jumpComponent.getStateVariableIndex());
		assertEquals(LOWER_BOUND, jumpComponent.getLowerIntegrationBound(TIME, 100.0, 0.2, 1.0), TOL);
		assertEquals(UPPER_BOUND, jumpComponent.getUpperIntegrationBound(TIME, 100.0, 0.2, 1.0), TOL);
	}

	@Test
	public void testLevyDensityIsNonNegativeAndPeaksAtMean() {
		final MertonJumpComponent jumpComponent = new MertonJumpComponent(
				LOWER_BOUND,
				UPPER_BOUND,
				INTENSITY,
				JUMP_MEAN,
				JUMP_STD_DEV
		);

		final double densityAtMean = jumpComponent.getLevyDensity(TIME, JUMP_MEAN);

		assertTrue(densityAtMean > 0.0);
		assertTrue(jumpComponent.getLevyDensity(TIME, JUMP_MEAN - 2.0 * JUMP_STD_DEV) >= 0.0);
		assertTrue(jumpComponent.getLevyDensity(TIME, JUMP_MEAN + 2.0 * JUMP_STD_DEV) >= 0.0);
		assertTrue(jumpComponent.getLevyDensity(TIME, LOWER_BOUND) >= 0.0);
		assertTrue(jumpComponent.getLevyDensity(TIME, UPPER_BOUND) >= 0.0);

		assertTrue(densityAtMean > jumpComponent.getLevyDensity(TIME, JUMP_MEAN - JUMP_STD_DEV));
		assertTrue(densityAtMean > jumpComponent.getLevyDensity(TIME, JUMP_MEAN + JUMP_STD_DEV));
	}

	@Test
	public void testLevyDensityIsSymmetricAroundJumpMean() {
		final MertonJumpComponent jumpComponent = new MertonJumpComponent(
				LOWER_BOUND,
				UPPER_BOUND,
				INTENSITY,
				JUMP_MEAN,
				JUMP_STD_DEV
		);

		final double[] offsets = new double[] {
				0.1 * JUMP_STD_DEV,
				0.5 * JUMP_STD_DEV,
				1.0 * JUMP_STD_DEV,
				2.0 * JUMP_STD_DEV
		};

		for(final double offset : offsets) {
			final double leftDensity = jumpComponent.getLevyDensity(TIME, JUMP_MEAN - offset);
			final double rightDensity = jumpComponent.getLevyDensity(TIME, JUMP_MEAN + offset);

			assertEquals(leftDensity, rightDensity, 1E-12);
		}
	}

	@Test
	public void testIntegratedLevyDensityRecoversJumpIntensity() {
		final MertonJumpComponent jumpComponent = new MertonJumpComponent(
				LOWER_BOUND,
				UPPER_BOUND,
				INTENSITY,
				JUMP_MEAN,
				JUMP_STD_DEV
		);

		final int numberOfSteps = 20000;
		final double recoveredMass = integrateTrapezoidal(jumpComponent, TIME, LOWER_BOUND, UPPER_BOUND, numberOfSteps);

		assertEquals(INTENSITY, recoveredMass, 1E-6);
	}

	private static double integrateTrapezoidal(
			final MertonJumpComponent jumpComponent,
			final double time,
			final double lowerBound,
			final double upperBound,
			final int numberOfSteps) {

		final double dx = (upperBound - lowerBound) / numberOfSteps;

		double integral = 0.0;
		double previousX = lowerBound;
		double previousValue = jumpComponent.getLevyDensity(time, previousX);

		for(int i = 1; i <= numberOfSteps; i++) {
			final double currentX = lowerBound + i * dx;
			final double currentValue = jumpComponent.getLevyDensity(time, currentX);

			integral += 0.5 * (previousValue + currentValue) * dx;

			previousX = currentX;
			previousValue = currentValue;
		}

		return integral;
	}
}