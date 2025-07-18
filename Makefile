CXX     := clang++
#LD	    := clang

CXXFLAGS += -std=c++23
#LDFLAGS += 

.PHONY: all clean shaders

all: shaders build/vk

clean:
	rm -rf build

shaders: build/vert.spv build/frag.spv

build/vk: build/main.o
	$(CXX) $(LDFLAGS) -lglfw -lvulkan $^ -o $@

build/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/vert.spv: vert.glsl
	glslc -fshader-stage=vert $< -o $@

build/frag.spv: frag.glsl
	glslc -fshader-stage=frag $< -o $@
