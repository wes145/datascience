from pwn import *
def extract_bytes_from_output(output):
    start_delimiter = b'----------------------------- START -------------------------------\n'
    end_delimiter = b'\n------------------------------ END'

    # Find the index of the start and end delimiters in the output
    start_index = output.find(start_delimiter)
    end_index = output.find(end_delimiter)

    # Extract the bytes between the start and end delimiters
    if start_index != -1 and end_index != -1:
        encrypted_bytes = output[start_index + len(start_delimiter):end_index]
        return encrypted_bytes
    else:
        return b''  

def send_and_receive(data_to_send):
    # Establish a connection to the server
    conn = remote('chals.t.cyberthon24.ctf.sg', 32011)

    # Set a timeout of 10 seconds
    conn.settimeout(10)

    # Receive the banner
    try:
        data = conn.recvuntil(b'END')
    except Timeout:
        print("Timeout while receiving banner")
        conn.close()
        return None

    # Send some data to be encrypted
    conn.send(data_to_send)
    conn.shutdown('send')

    # Receive the encrypted data
    try:
        data = conn.recvuntil(b'END')
    except Timeout:
        print("Timeout while receiving encrypted data")
        conn.close()
        return None

    # Close the connection
    conn.close()

    return extract_bytes_from_output(data)
byte_to_encrypted = {}
data_to_send = bytes(range(256))
received_data = send_and_receive(data_to_send)

# Split up the received data into 256-byte chunks
# Calculate the length of each chunk
chunk_length = len(received_data) // 256

# Split the received data into chunks of this length
chunks = [received_data[i:i+chunk_length] for i in range(0, len(received_data), chunk_length)]

# Create a dictionary where the keys are the bytes that were sent and the values are the corresponding chunks
byte_to_ciphertext = {bytes([i]): chunk for i, chunk in enumerate(chunks)}

# Print the dictionary
for key, value in byte_to_ciphertext.items():
    print(f'{key}: {value}')
def decrypt_ciphertext(ciphertext, byte_to_ciphertext):
    # Reverse the byte_to_ciphertext dictionary
    ciphertext_to_byte = {v: k for k, v in byte_to_ciphertext.items()}

    # Calculate the length of each chunk
    chunk_length = len(list(byte_to_ciphertext.values())[0])

    # Split the ciphertext into chunks of this length
    chunks = [ciphertext[i:i+chunk_length] for i in range(0, len(ciphertext), chunk_length)]

    # Replace each chunk with the corresponding byte
    decrypted_message = b''.join(ciphertext_to_byte[chunk] for chunk in chunks)

    return decrypted_message
# Read the ciphertext from a file
with open(r"C:\\Users\\wesle\\Downloads\\flag.txt (7).encrypted", 'rb') as f:
    ciphertext = f.read()

# Decrypt the ciphertext
decrypted_message = decrypt_ciphertext(ciphertext, byte_to_ciphertext)

# Print the decrypted message
print(decrypted_message)
# Cycle through the 256 possible bytes
# for i in range(256):
#     data_to_send = bytes([i])
#     received_data = send_and_receive(data_to_send)
#     byte_to_encrypted[data_to_send] = received_data

# # Print the dictionary
# for key, value in byte_to_encrypted.items():
#     print(f'{key}: {value}')