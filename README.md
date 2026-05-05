# finmath-lib-new-fdm

`finmath-lib-new-fdm` is a standalone development repository for a rewritten finite-difference engine for the Java [`finmath-lib`](https://github.com/finmath/finmath-lib) ecosystem. The code focuses on finite-difference, PIDE, and ADI pricing methods for equity, jump, stochastic-volatility, multi-asset, and one-factor interest-rate products.

The current codebase should be read as an actively evolving FDM engine rather than a published stable API. The Maven artifact is currently `net.finmath:fdm:0.0.1-SNAPSHOT`, targets Java 11, and depends on `finmath-lib` plus plotting/test support.

## Current status

- **Coverage matrix:** updated after commit `196a3c33de466df25c518ad8a9c4490c5c0546d9`.
- **Repo head inspected for this README:** commit `16e85650ba7ea2496fd0c593c073bbc9a36d1811` adds plotting utilities and a Black-Scholes surface demo for price, delta, gamma, and theta surfaces. This is newer than the coverage-matrix anchor and does not change the product coverage table.
- **Primary focus:** reusable FDM/PIDE/ADI infrastructure, model/product separation, event handling, barriers, discrete monitoring, and regression-tested product coverage.
- **License:** Apache License 2.0, following the surrounding `net.finmath.*` library convention.

## Highlights

- 1D finite-difference theta-method solver stack for diffusion models.
- 1D PIDE support for jump models such as Merton and Variance Gamma.
- 2D/3D ADI-style solvers for stochastic-volatility, jump-stochastic-volatility, Asian, and multi-asset problems.
- Product-level support for European, Bermudan, American, barrier, digital, Asian, basket, shout, and swing-style contracts.
- Event-condition layer for exercise, monitoring, settlement, and path/state-dependent features.
- One-factor Hull-White rate-domain baseline with bonds, swaps, and swaptions.
- Surface interpolation, finite-difference Greek utilities, and JavaFX-based 3D plotting demos for diagnostics.

## Repository layout

```text
src/main/java/net/finmath/finitedifference/
├── grids/                         # Uniform, exponential, hyperbolic-sine, and space-time grids
├── boundaries/                    # Generic boundary-condition abstractions
├── solvers/                       # 1D theta/PIDE solvers, SOR/Thomas solvers, solver factory
├── solvers/adi/                   # 2D/3D ADI solvers and stencil utilities
├── assetderivativevaluation/
│   ├── models/                    # Black-Scholes, Bachelier, CEV, Heston, SABR, jump models
│   ├── products/                  # Equity/exotic product definitions and event products
│   └── boundaries/                # Product/model-specific active boundary providers
├── interestrate/
│   ├── models/                    # Hull-White 1D model support
│   ├── products/                  # Bonds, swap legs, swaps, swaptions
│   └── boundaries/                # Rate-product boundary logic
└── utilities/                     # Interpolation, Greeks, surface views, plot-data helpers
```

## Build and test

```bash
git clone https://github.com/AlessandroGnoatto/finmath-lib-new-fdm.git
cd finmath-lib-new-fdm
mvn test
```

The plotting demos use JavaFX/Swing-style interactive windows and may require a graphical environment. For headless CI, keep plotting demos separate from numerical regression tests.

## Minimal usage sketch

```java
final SpaceTimeDiscretization discretization = /* create grid and time discretization */;

final FDMBlackScholesModel model = new FDMBlackScholesModel(
        initialSpot,
        riskFreeRate,
        dividendYield,
        volatility,
        discretization
);

final EuropeanOption callOption = new EuropeanOption(
        maturity,
        strike,
        CallOrPut.CALL
);

final double[][] values = callOption.getValues(model);

final FiniteDifferenceSurfaceView valueView =
        FiniteDifferenceSurfaceView.of(discretization, values);

final double valueAtSpot = valueView.interpolate(
        0.0,
        maturity,
        initialSpot
);
```

## Equity, jump, and multi-asset coverage

| Product / payoff family | Model coverage | Exercise styles | Monitoring | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Vanilla call/put | Black-Scholes, Bachelier, CEV, Heston, SABR | European, Bermudan, American | Continuous | Implemented | Core vanilla option stack across five diffusive equity models. |
| Jump vanilla call/put | Black-Scholes (Merton only) | European | PIDE / jump | Partial | Merton and Variance Gamma in 1D; Bates in 2D. The jump columns are not generic across all payoff families. |
| Cash/asset digital call/put | Black-Scholes, Bachelier, CEV, Heston, SABR | European, Bermudan, American | Continuous | Implemented | DigitalOption supports cash-or-nothing and asset-or-nothing variants. |
| Single-barrier vanilla | Black-Scholes, Bachelier, CEV, Heston, SABR | European, Bermudan, American | Continuous + discrete | Implemented | BarrierOption supports knock-in/out and discrete monitoring patterns. |
| Single-barrier digital | Black-Scholes, Bachelier, CEV, Heston, SABR | European, Bermudan, American | Continuous + discrete | Implemented | DigitalBarrierOption supports cash/asset digitals, knock-in/out, and discrete monitoring. |
| Touch / no-touch | Black-Scholes, Bachelier, CEV, Heston, SABR | European | Continuous / settlement events | Partial | TouchOption includes one-touch/no-touch semantics; early-exercise completion remains a cleanup item. |
| Asian arithmetic | Black-Scholes, Bachelier, CEV, Heston, SABR | European | Fixing schedule | Implemented | AsianOption spans the five diffusive equity model families. |
| Double-barrier vanilla | Black-Scholes, Bachelier, CEV, Heston, SABR | European, Bermudan, American | Continuous + discrete | Implemented | DoubleBarrierOption covers vanilla KI/KO and discrete monitoring under the current barrier stack. |
| Double-barrier binary KO | Black-Scholes, Bachelier, CEV, Heston, SABR | European, Bermudan, American | Continuous + discrete | Implemented | Latest commit adds discrete monitoring for DoubleBarrierBinaryOption. |
| Double-barrier binary KI | Black-Scholes, Bachelier, CEV, Heston, SABR | European, Bermudan, American | Continuous + discrete | Implemented | Discrete KI uses cached activated cash vectors and vector event conditions. |
| Double-barrier binary KIKO | Black-Scholes, Bachelier, CEV, Heston, SABR | European, Bermudan, American | Continuous + discrete | Implemented | Discrete KIKO is regression-tested through KIKO+KOKI ~= KI identity. |
| Double-barrier binary KOKI | Black-Scholes, Bachelier, CEV, Heston, SABR | European, Bermudan, American | Continuous + discrete | Implemented | Discrete KOKI is regression-tested across BS, CEV, Bachelier, Heston, and SABR. |
| Shout call/put | Black-Scholes, Bachelier, CEV, Heston, SABR | European | Continuous shout rights | Implemented | Finite multi-shout reset rule K*=S across five diffusive models. |
| Swing fixed strike | Black-Scholes, Bachelier, CEV, Heston, SABR | Bermudan | Discrete exercise/control | Implemented | Generalized swing with local/global quantity constraints. |
| Swing floating strike | Black-Scholes, Bachelier, CEV, Heston, SABR | Bermudan | Discrete fixing/control | Implemented | Arithmetic floating strike with fix-then-exercise ordering. |
| 2D basket / spread / exchange | Black-Scholes (2D multi-asset) | European | Continuous | Implemented | BasketOption covers linear basket, spread, and exchange special cases under 2D Black-Scholes. |
| Best-of / worst-of | Black-Scholes (2D multi-asset) | European | Continuous | Implemented | BestOfOption and WorstOfOption under 2D multi-asset Black-Scholes. |
| Digital basket | Black-Scholes (2D multi-asset) | European | Continuous | Implemented | DigitalBasketOption under 2D multi-asset Black-Scholes. |

## Rates coverage: Hull-White 1D

The latest product-coverage update did not change the rates stack. Hull-White remains the first rate-domain PDE baseline.

| Capability / product variant | Hull-White 1D | Exercise styles | Validation | Notes |
| --- | --- | --- | --- | --- |
| Zero-coupon bond | Yes | N/A | Analytic regression | Closed-form discount-bond valuation under 1D Hull-White. |
| Fixed-coupon bond | Yes | N/A | Discounted-cashflow regression | Schedule-based bond with coupon/event handling. |
| Option on bond | Yes | European | Analytic Hull-White bond-option regression | European option on deterministic-cashflow bond with exact Hull-White boundary logic. |
| Swap annuity helper | Yes | N/A | Deterministic identity checks | Reduced-scope fixed-leg annuity helper. |
| Swap leg fixed/floating | Yes | N/A | Monte Carlo Hull-White regression | Direct grid valuation from discount bonds and forward rates. |
| Swap | Yes | N/A | Monte Carlo Hull-White regression | Receiver leg minus payer leg with par-rate/off-market tests. |
| Swaption payer/receiver | Yes | European, Bermudan, American (Approximation on solver grid) | Monte Carlo Hull-White regression | Unified Exercise-driven Swaption class; American is grid-based approximation. |

## Validation and evidence notes

| Area checked | Result | Evidence / file path |
| --- | --- | --- |
| Coverage workbook anchor | discrete monitoring for DoubleBarrierBinaryOption | commit 196a3c33de466df25c518ad8a9c4490c5c0546d9 |
| Double-barrier cash binary product | Continuous and discrete monitoring via event conditions; thread-local activated state for KI/KIKO/KOKI | src/main/java/net/finmath/finitedifference/assetderivativevaluation/products/DoubleBarrierBinaryOption.java |
| Discrete monitoring helpers | Monitoring validation and time-grid refinement available | src/main/java/net/finmath/finitedifference/assetderivativevaluation/products/internal/DiscreteMonitoringSupport.java |
| Solver event layer | FDMThetaMethod1D applies FiniteDifferenceEquityEventProduct / rate-product event conditions at event times | src/main/java/net/finmath/finitedifference/solvers/FDMThetaMethod1D.java |
| Discrete binary barrier regression | Covers BS, CEV, Bachelier, Heston, SABR; checks KI+KO, KIKO+KOKI, bounds, and discrete-vs-continuous ordering | src/test/java/net/finmath/finitedifference/assetderivativevaluation/products/DoubleBarrierBinaryOptionDiscreteMonitoringRegressionTest.java |
| Still absent in repo search | CallableBond, PuttableBond, CancelableSwap, G2, CIR, HestonHullWhite, storage, VPP | No matching implementation files found in connected-repo search. |

## Known limitations and open areas

- Jump-model coverage is not generic across all payoff families. Current jump coverage is concentrated in 1D Merton/Variance-Gamma vanilla pricing and 2D Bates-style pricing.
- Multi-asset products are currently centered on 2D Black-Scholes basket/spread/exchange/best-of/worst-of/digital-basket use cases.
- The rates stack is currently limited to 1D Hull-White. G2, CIR, hybrid Heston-Hull-White, callable bonds, puttable bonds, cancelable swaps, storage, and VPP-style products are not implemented in the connected coverage matrix.
- American swaption support is marked as an approximation on the solver grid.
- Touch/no-touch products are partially covered: one-touch/no-touch semantics are present, while early-exercise completion remains a cleanup item.
- Plotting utilities are useful for diagnostics but should remain isolated from production dependencies and headless numerical regression tests where possible.

## Coding conventions

The project follows the surrounding `finmath-lib` style: Java code under `net.finmath.*`, Eclipse-inspired formatting conventions, and preference for clear mathematical structure over overly compact implementation. Long formula-heavy lines may occur where they improve readability of numerical expressions.

## License

The code under `net.finmath.*` is distributed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0.html), unless otherwise explicitly stated.
