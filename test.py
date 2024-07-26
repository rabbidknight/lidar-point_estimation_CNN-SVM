import struct

# Example byte array for a single 'x' value
bytes_for_x = [193, 34, 3, 68]

# Convert bytes to a float
x_value = struct.unpack('<f', bytearray(bytes_for_x))[0]

print(x_value)
