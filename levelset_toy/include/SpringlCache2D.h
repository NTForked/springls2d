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

#ifndef INCLUDE_SPRINGLCACHE2D_H_
#define INCLUDE_SPRINGLCACHE2D_H_
#include "Manifold2D.h"
#include <mutex>
#include <map>
#include <set>
namespace aly {
	class CacheElement {
	protected:
		bool loaded;
		bool writeOnce;
		std::string contourFile;
		std::shared_ptr<Manifold2D> contour;

		std::mutex accessLock;
	public:
		bool isLoaded() {
			std::lock_guard<std::mutex> lockMe(accessLock);
			return loaded;
		}
		CacheElement():loaded(false), writeOnce(true){
		}
		~CacheElement();
		std::string getFile() const {
			return contourFile;
		}
		void load();
		void unload();
		void set(const Manifold2D& springl);
		std::shared_ptr<Manifold2D> getContour();
	};
	struct CacheCompare {
		inline bool operator() (const std::pair<uint64_t, int>& lhs, const std::pair<uint64_t, int>& rhs) const {
			return lhs.first < rhs.first;
		}
	};
	class SpringlCache2D {
	protected:
		std::map<int, std::shared_ptr<CacheElement>> cache;
		std::set<std::pair<uint64_t, int>, CacheCompare> loadedList;
		std::mutex accessLock;
		int maxElements = 32;
		uint64_t counter = 0;
	public:
		std::shared_ptr<CacheElement> set(int frame, const Manifold2D& springl);
		std::shared_ptr<CacheElement> get(int frame);
		void clear();
	};

}

#endif /* INCLUDE_SpringlCache_H_ */
