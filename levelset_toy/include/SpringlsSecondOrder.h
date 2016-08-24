/*
 * Copyright(C) 2016, Blake C. Lucas, Ph.D. (img.science@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef INCLUDE_SPRINGLSSECONDORDER_H_
#define INCLUDE_SPRINGLSSECONDORDER_H_
#include "ActiveContour2D.h"
#include "Simulation.h"
#include "SpringlCache2D.h"
#include "ContourShaders.h"
#include "AlloyLocator.h"
#include "SpringLevelSet2D.h"
namespace aly {
	class SpringlsSecondOrder : public SpringLevelSet2D{

	protected:
		virtual void computeForce(size_t idx, float2& p1, float2& p2, float2& p) override;


	public:

		SpringlsSecondOrder(const std::shared_ptr<SpringlCache2D>& cache = nullptr);

	};
}

#endif /* INCLUDE_SpringlsSecondOrder_H_ */
