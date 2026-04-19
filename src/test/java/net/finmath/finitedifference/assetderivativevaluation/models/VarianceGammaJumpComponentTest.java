package net.finmath.finitedifference.assetderivativevaluation.models;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Unit tests for {@link VarianceGammaJumpComponent}.
 *
 * @author Alessandro Gnoatto
 */
public class VarianceGammaJumpComponentTest {

	private static final double LOWER_BOUND = -2.0;
	private static final double UPPER_BOUND = 2.0;

	private static final double C = 2.5;
	private static final double G = 6.0;
	private static final double M = 8.0;

	private static final double SIGMA = 0.30;
	private static final double NU = 0.20;
	private static final double THETA = -0.10;

	private static final double TIME = 0.5;
	private static final double TOL = 1E-12;

	@Test
	public void testConstructorMetadataAndParameters() {
		final VarianceGammaJumpComponent jumpComponent = new VarianceGammaJumpComponent(
				LOWER_BOUND,
				UPPER_BOUND,
				C,
				G,
				M
		);

		assertEquals(0, jumpComponent.getStateVariableIndex());
		assertTrue(!jumpComponent.isStateDependent());
		assertTrue(!jumpComponent.isFiniteActivity());
		assertTrue(jumpComponent.isFiniteVariation());

		assertEquals(LOWER_BOUND, jumpComponent.getLowerIntegrationBound(TIME, 100.0), TOL);
		assertEquals(UPPER_BOUND, jumpComponent.getUpperIntegrationBound(TIME, 100.0), TOL);

		assertEquals(C, jumpComponent.getC(), TOL);
		assertEquals(G, jumpComponent.getG(), TOL);
		assertEquals(M, jumpComponent.getM(), TOL);
	}

	@Test
	public void testCustomStateVariableIndex() {
		final int stateVariableIndex = 2;

		final VarianceGammaJumpComponent jumpComponent = new VarianceGammaJumpComponent(
				stateVariableIndex,
				LOWER_BOUND,
				UPPER_BOUND,
				C,
				G,
				M
		);

		assertEquals(stateVariableIndex, jumpComponent.getStateVariableIndex());
		assertEquals(LOWER_BOUND, jumpComponent.getLowerIntegrationBound(TIME, 100.0, 0.2, 1.0), TOL);
		assertEquals(UPPER_BOUND, jumpComponent.getUpperIntegrationBound(TIME, 100.0, 0.2, 1.0), TOL);
	}

	@Test
	public void testLevyDensityIsPositiveOnBothSidesOfZero() {
		final VarianceGammaJumpComponent jumpComponent = new VarianceGammaJumpComponent(
				LOWER_BOUND,
				UPPER_BOUND,
				C,
				G,
				M
		);

		final double negativeDensity = jumpComponent.getLevyDensity(TIME, -0.25);
		final double positiveDensity = jumpComponent.getLevyDensity(TIME, 0.25);

		assertTrue(negativeDensity > 0.0);
		assertTrue(positiveDensity > 0.0);
	}

	@Test
	public void testLevyDensityMatchesFormulaOnBothSides() {
		final VarianceGammaJumpComponent jumpComponent = new VarianceGammaJumpComponent(
				LOWER_BOUND,
				UPPER_BOUND,
				C,
				G,
				M
		);

		final double positiveJump = 0.4;
		final double negativeJump = -0.3;

		final double expectedPositiveDensity = C * Math.exp(-M * positiveJump) / positiveJump;
		final double expectedNegativeDensity = C * Math.exp(-G * Math.abs(negativeJump)) / Math.abs(negativeJump);

		assertEquals(expectedPositiveDensity, jumpComponent.getLevyDensity(TIME, positiveJump), TOL);
		assertEquals(expectedNegativeDensity, jumpComponent.getLevyDensity(TIME, negativeJump), TOL);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testLevyDensityThrowsAtZero() {
		final VarianceGammaJumpComponent jumpComponent = new VarianceGammaJumpComponent(
				LOWER_BOUND,
				UPPER_BOUND,
				C,
				G,
				M
		);

		jumpComponent.getLevyDensity(TIME, 0.0);
	}

	@Test
	public void testDensityExplodesNearZero() {
		final VarianceGammaJumpComponent jumpComponent = new VarianceGammaJumpComponent(
				LOWER_BOUND,
				UPPER_BOUND,
				C,
				G,
				M
		);

		final double densityAtMinusOneTenth = jumpComponent.getLevyDensity(TIME, -1E-1);
		final double densityAtMinusOneHundredth = jumpComponent.getLevyDensity(TIME, -1E-2);

		final double densityAtPlusOneTenth = jumpComponent.getLevyDensity(TIME, 1E-1);
		final double densityAtPlusOneHundredth = jumpComponent.getLevyDensity(TIME, 1E-2);

		assertTrue(densityAtMinusOneHundredth > densityAtMinusOneTenth);
		assertTrue(densityAtPlusOneHundredth > densityAtPlusOneTenth);
	}

	@Test
	public void testOfSigmaNuThetaMatchesDerivedCGMParameters() {

		final VarianceGammaJumpComponent jumpComponent =
				VarianceGammaJumpComponent.ofSigmaNuTheta(
						LOWER_BOUND,
						UPPER_BOUND,
						SIGMA,
						NU,
						THETA
				);

		final double sigmaSquared = SIGMA * SIGMA;
		final double radical = Math.sqrt(THETA * THETA + 2.0 * sigmaSquared / NU);

		final double expectedC = 1.0 / NU;
		final double expectedG = (radical + THETA) / sigmaSquared;
		final double expectedM = (radical - THETA) / sigmaSquared;

		assertEquals(expectedC, jumpComponent.getC(), TOL);
		assertEquals(expectedG, jumpComponent.getG(), TOL);
		assertEquals(expectedM, jumpComponent.getM(), TOL);
	}

	@Test
	public void testFactoryAndDirectConstructorGiveSameDensity() {

		final double sigmaSquared = SIGMA * SIGMA;
		final double radical = Math.sqrt(THETA * THETA + 2.0 * sigmaSquared / NU);

		final double derivedC = 1.0 / NU;
		final double derivedG = (radical + THETA) / sigmaSquared;
		final double derivedM = (radical - THETA) / sigmaSquared;

		final VarianceGammaJumpComponent fromFactory =
				VarianceGammaJumpComponent.ofSigmaNuTheta(
						LOWER_BOUND,
						UPPER_BOUND,
						SIGMA,
						NU,
						THETA
				);

		final VarianceGammaJumpComponent fromDirectConstructor =
				new VarianceGammaJumpComponent(
						LOWER_BOUND,
						UPPER_BOUND,
						derivedC,
						derivedG,
						derivedM
				);

		final double[] jumpSizes = new double[] { -0.7, -0.2, -0.05, 0.05, 0.2, 0.7 };

		for(final double jumpSize : jumpSizes) {
			assertEquals(
					fromDirectConstructor.getLevyDensity(TIME, jumpSize),
					fromFactory.getLevyDensity(TIME, jumpSize),
					TOL
			);
		}
	}
}