#include <cstring>

#include <ImageMagick-7/Magick++.h>

#include "macros/assert.hpp"
#include "pixelbuffer.hpp"

auto load_texture_imagemagick(Magick::Image&& image) -> PixelBuffer {
    const auto width  = image.columns();
    const auto height = image.rows();
    auto       data   = std::vector<std::byte>(width * height * 4);
    image.write(0, 0, width, height, "RGBA", Magick::CharPixel, data.data());

    return PixelBuffer{width, height, std::move(data)};
}

auto PixelBuffer::from_raw(const size_t width, const size_t height, const std::byte* const buffer) -> PixelBuffer {
    const auto len = size_t(width * height * 4);

    auto data = std::vector<std::byte>(len);
    std::memcpy(data.data(), buffer, len);

    return PixelBuffer{width, height, std::move(data)};
}

auto PixelBuffer::from_file(const char* const file) -> std::optional<PixelBuffer> {
    try {
        return load_texture_imagemagick(Magick::Image(file));
    } catch(const Magick::Exception& e) {
        bail("imagemagick error: {}", e.what());
    }
}

auto PixelBuffer::from_blob(const std::byte* const data, const size_t size) -> std::optional<PixelBuffer> {
    try {
        auto blob = Magick::Blob(data, size);
        return load_texture_imagemagick(Magick::Image(blob));
    } catch(const Magick::Exception& e) {
        bail("imagemagick error: {}", e.what());
    }
}

auto PixelBuffer::from_blob(const std::span<const std::byte> buffer) -> std::optional<PixelBuffer> {
    return from_blob(buffer.data(), buffer.size());
}
