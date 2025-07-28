CXX     := clang++
#LD	    := clang

CXXFLAGS += -std=c++23
#LDFLAGS += 

.PHONY: all clean shaders

all: shaders build/vk build/vk2

clean:
	rm -rf build

shaders: build/vert.spv build/frag.spv

build/vk: build/vk.o  build/main.o build/vert.spv build/frag.spv
	$(CXX) $(LDFLAGS) -lglfw -lvulkan $^ -o $@

build/vk2: 	build/vk.o \
			build/main2.o \
			build/pixelbuffer.o \
			build/buffer.o \
			build/image.o \
			build/command.o \
			build/tiny_obj_loader.o \
			build/vert.spv \
			build/frag.spv \
			build/tiny_obj_loader.h \
			build/viking_room.obj \
			build/viking_room.png
	$(CXX) $(LDFLAGS) -lglfw -lvulkan $(shell pkg-config -libs Magick++) $(filter %.o,$^) -o $@

build/tiny_obj_loader.h:
	curl -L -o $@ "https://raw.githubusercontent.com/tinyobjloader/tinyobjloader/refs/heads/release/tiny_obj_loader.h"

build/viking_room.obj:
	curl -L -o $@ "https://vulkan-tutorial.com/resources/viking_room.obj"

build/viking_room.png:
	curl -L -o $@ "https://vulkan-tutorial.com/resources/viking_room.png"

build/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(shell pkg-config -cflags Magick++) -c $< -o $@

build/vert.spv: vert.glsl
	glslc -fshader-stage=vert $< -o $@

build/frag.spv: frag.glsl
	glslc -fshader-stage=frag $< -o $@
