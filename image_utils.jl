module ImageUtils

using FileIO
using ImageIO
using ColorTypes
using ImageCore

"""
	image_to_tokens(img::AbstractArray{<:Real,3}; normalize_xy::Bool=true, rgb_to_unit::Bool=true, as_columns::Bool=true) -> Matrix{Float32}

	Converts an image array of size (H, W, 3) into tokens containing
	[x, y, r, g, b] per pixel.

	- When `normalize_xy = true` (default), x and y are mapped to [-1, 1] with top-left = (-1, -1).
	  If `false`, x and y are 1-based pixel indices: x ∈ [1, W], y ∈ [1, H].
	- When `rgb_to_unit = true` (default), RGB are scaled to [-1, 1] using divisor 1 if max≤1,
	  255 if max≤255, else max.
	- When `as_columns = true`, output has shape (5, N) with each column = [x, y, r, g, b].
	  Otherwise, shape is (N, 5) with each row = [x, y, r, g, b].
"""
function image_to_tokens(img::AbstractArray{<:Real,3}; normalize_xy::Bool=true, rgb_to_unit::Bool=true, as_columns::Bool=true)::Matrix{Float32}
	(size(img, 3) == 3) || throw(ArgumentError("img must have 3 channels in the last dimension (H×W×3)."))
	H, W, _ = size(img)
	N = H * W

	# Determine RGB scaling
	max_val = float(maximum(img))
	rgb_scale::Float32 = if rgb_to_unit
		# Scale to [-1, 1]: divisor = 1 if max≤1; 255 if max≤255; else max
		divisor::Float32 = (max_val <= 1.0f0) ? 1.0f0 : ((max_val <= 255.0f0) ? 255.0f0 : Float32(max_val))
		2.0f0 / divisor  # Scale to [-1, 1] instead of [0, 1]
	else
		max_val <= 1.0 ? 255.0f0 : 1.0f0
	end


	# Prepare output
	if as_columns
		tokens = Matrix{Float32}(undef, 5, N)
	else
		tokens = Matrix{Float32}(undef, N, 5)
	end

	@inline normalize_index(k::Int, K::Int)::Float32 = K == 1 ? 0.0f0 : -1.0f0 + 2.0f0 * Float32(k - 1) / Float32(K - 1)

	n = 0
	@inbounds for j in 1:W
		# x coordinate
		x = normalize_xy ? normalize_index(j, W) : Float32(j)
		for i in 1:H
			# y coordinate
			y = normalize_xy ? normalize_index(i, H) : Float32(i)
			n += 1
			r = Float32(img[i, j, 1]) * rgb_scale - 1.0f0  # Shift to [-1, 1]
			g = Float32(img[i, j, 2]) * rgb_scale - 1.0f0  # Shift to [-1, 1]
			b = Float32(img[i, j, 3]) * rgb_scale - 1.0f0  # Shift to [-1, 1]
			if as_columns
				tokens[1, n] = x
				tokens[2, n] = y
				tokens[3, n] = r
				tokens[4, n] = g
				tokens[5, n] = b
			else
				tokens[n, 1] = x
				tokens[n, 2] = y
				tokens[n, 3] = r
				tokens[n, 4] = g
				tokens[n, 5] = b
			end
		end
	end

	return tokens
end

"""
	image_to_tokens(path::AbstractString; kwargs...) -> Matrix{Float32}

	Loads an image (e.g., PNG) from `path` and returns tokens with
	[x, y, r, g, b] per pixel. See keyword arguments in the array method
	for behavior. Defaults: normalized x,y∈[-1,1] and RGB∈[-1,1], with
	columns storing pixels (`as_columns = true`).
"""
function image_to_tokens(path::AbstractString; kwargs...)::Matrix{Float32}
	img_loaded = try
		FileIO.load(path)
	catch err
		throw(ArgumentError("Failed to load image at '$path': $(err)"))
	end
	img_hwc = _to_hwc3(img_loaded)
	return image_to_tokens(img_hwc; kwargs...)
end

# Internal: convert loaded image to H×W×3 Float32 array
function _to_hwc3(img_loaded)
	if ndims(img_loaded) == 2 && eltype(img_loaded) <: Colorant
		ch = channelview(img_loaded) # C × H × W
		(size(ch, 1) == 3) || throw(ArgumentError("Loaded image must have 3 color channels (RGB)."))
		H = size(ch, 2); W = size(ch, 3)
		out = Array{Float32}(undef, H, W, 3)
		@inbounds for j in 1:W
			for i in 1:H
				out[i, j, 1] = Float32(ch[1, i, j])
				out[i, j, 2] = Float32(ch[2, i, j])
				out[i, j, 3] = Float32(ch[3, i, j])
			end
		end
		return out
	elseif ndims(img_loaded) == 3
		# Try to interpret as H×W×3 already
		(size(img_loaded, 3) == 3) || throw(ArgumentError("Loaded image must have 3 channels in the last dimension (H×W×3)."))
		return Float32.(img_loaded)
	else
		throw(ArgumentError("Unsupported loaded image shape: ndims=$(ndims(img_loaded))"))
	end
end

"""
	png_to_data_img(path::AbstractString)::Matrix{Float32}

	Convenience wrapper for `image_to_tokens` that returns a (5, N) matrix
	with columns = [x, y, r, g, b], with normalized coordinates x,y ∈ [-1,1]
	and RGB scaled to [-1,1].
"""
function png_to_data_img(path::AbstractString)::Matrix{Float32}
	return image_to_tokens(path; normalize_xy=true, rgb_to_unit=true, as_columns=true)
end

export image_to_tokens, png_to_data_img

end # module


