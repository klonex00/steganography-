from OpenSSL import crypto
import os

def generate_self_signed_cert(cert_file, key_file, common_name):
    """Generate a self-signed certificate and private key."""
    # Generate key
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)

    # Generate certificate
    cert = crypto.X509()
    cert.get_subject().C = "US"  # Country
    cert.get_subject().ST = "State"  # State
    cert.get_subject().L = "City"  # Location
    cert.get_subject().O = "Organization"  # Organization
    cert.get_subject().OU = "Organizational Unit"  # Organizational Unit
    cert.get_subject().CN = common_name  # Common Name
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for one year
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')

    # Write certificate
    with open(cert_file, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))

    # Write private key
    with open(key_file, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

def main():
    # Create certificates directory if it doesn't exist
    os.makedirs('certificates', exist_ok=True)

    # Generate server certificate and key
    print("Generating server certificate and key...")
    generate_self_signed_cert(
        'certificates/server.crt',
        'certificates/server.key',
        'localhost'
    )

    # Generate client certificate and key
    print("Generating client certificate and key...")
    generate_self_signed_cert(
        'certificates/client.crt',
        'certificates/client.key',
        'client'
    )

    print("Certificates and keys generated successfully!")
    print("Files created:")
    print("  - certificates/server.crt")
    print("  - certificates/server.key")
    print("  - certificates/client.crt")
    print("  - certificates/client.key")

if __name__ == "__main__":
    main() 