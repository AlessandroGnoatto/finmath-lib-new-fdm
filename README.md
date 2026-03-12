About finmath lib
==========

****************************************

**Mathematical Finance Library: Algorithms and methodologies related to mathematical finance.**

****************************************

[![Twitter](https://img.shields.io/twitter/url/https/github.com/finmath/finmath-lib.svg?style=social)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Ffinmath%2Ffinmath-lib)
[![GitHub license](https://img.shields.io/github/license/finmath/finmath-lib.svg)](https://github.com/finmath/finmath-lib/blob/master/LICENSE.txt)
[![Latest release](https://img.shields.io/github/release/finmath/finmath-lib.svg)](https://github.com/finmath/finmath-lib/releases/latest)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/net.finmath/finmath-lib/badge.svg)](https://maven-badges.herokuapp.com/maven-central/net.finmath/finmath-lib)
[![Build Status](https://travis-ci.org/finmath/finmath-lib.svg?branch=master)](https://travis-ci.org/finmath/finmath-lib)
[![javadoc](https://javadoc.io/badge2/net.finmath/finmath-lib/javadoc.svg)](https://javadoc.io/doc/net.finmath/finmath-lib)

---



**Project home page: http://finmath.net/finmath-lib**

This project represents a full rewrite of the finite difference method engine for the finmath java library. It is being developed as a separate project in view of a future integration in the library.



License
-------

The code of "finmath lib" and "finmath experiments" (packages
`net.finmath.*`) are distributed under the [Apache License version
2.0][], unless otherwise explicitly stated.
 

  [finmath lib Project documentation]: http://finmath.net/finmath-lib/ 
  [finmath lib API documentation]: http://finmath.net/finmath-lib/apidocs/
  [finmath.net special topics]: http://www.finmath.net/topics
  [Apache License version 2.0]: http://www.apache.org/licenses/LICENSE-2.0.html




Coding Conventions
-------------------------------------

We follow losely the Eclipse coding conventions, which are a minimal modification of the original Java coding conventions. See https://wiki.eclipse.org/Coding_Conventions

We deviate in some places:

-   We allow for long code lines. Some coding conventions limit the length of a line to something like 80 characters (like FORTRAN did in the 70'ies). Given widescreen monitors we believe that line wrapping makes code much harder to read than code with long(er) lines.

-	We usually do not make a space after statements like `íf`, `for`. We interpret `íf` and `for` as functions and for functions and methods we do not have a space between the name and the argument list either. That is, we write

    if(condition) {
      // code
    }

