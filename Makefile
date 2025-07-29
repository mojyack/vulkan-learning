CC 		:= clang
CXX     := clang++
#LD	    := clang

CXXFLAGS += -std=c++23 -I/tmp/rootfs/include -Ibuild
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
			build/xdg-shell.o \
			build/xdg-shell.h \
			build/towl-display.o \
			build/towl-registry.o \
			build/towl-interface.o \
			build/towl-compositor.o \
			build/towl-xdg-wm-base.o \
			build/vert.spv \
			build/frag.spv \
			build/tiny_obj_loader.h \
			build/viking_room.obj \
			build/viking_room.png
	$(CXX) $(LDFLAGS) -lglfw -lvulkan \
		$(shell pkg-config -libs Magick++) \
		$(shell pkg-config -libs wayland-client) \
		$(filter %.o,$^) -o $@

build/tiny_obj_loader.h:
	curl -L -o $@ "https://raw.githubusercontent.com/tinyobjloader/tinyobjloader/refs/heads/release/tiny_obj_loader.h"

build/viking_room.obj:
	curl -L -o $@ "https://vulkan-tutorial.com/resources/viking_room.obj"

build/viking_room.png:
	curl -L -o $@ "https://vulkan-tutorial.com/resources/viking_room.png"

build/xdg-shell.c:
	wayland-scanner public-code $(shell pkg-config --variable=pkgdatadir wayland-protocols)/stable/xdg-shell/xdg-shell.xml $@

build/xdg-shell.h:
	wayland-scanner client-header $(shell pkg-config --variable=pkgdatadir wayland-protocols)/stable/xdg-shell/xdg-shell.xml $@

build/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(shell pkg-config -cflags Magick++) -c $< -o $@

build/towl-%.o: towl/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/xdg-shell.o: build/xdg-shell.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

build/vert.spv: vert.glsl
	glslc -fshader-stage=vert $< -o $@

build/frag.spv: frag.glsl
	glslc -fshader-stage=frag $< -o $@
