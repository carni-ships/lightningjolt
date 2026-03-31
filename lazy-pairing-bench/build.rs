fn main() {
    println!("cargo:rustc-link-search=native=/tmp/constantine");
    println!("cargo:rustc-link-lib=static=ctt_bn254_pairing");
}
