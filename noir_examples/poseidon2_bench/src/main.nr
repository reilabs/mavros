use poseidon2::bn254::perm;

fn main(x: u32, y: u32, z: u32, w: u32) {
  let x = x as Field;
  let y = y as Field;
  let z = z as Field;
  let w = w as Field;

  //let r = perm::x5_2([x,x]);
  //let r = perm::x5_3([x,x,x]);
  let r = perm::x5_4([x,y,z,w]);
  //let r = perm::x5_8([x,x,x,x,x,x,x,x]);
  //let r = perm::x5_16([x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x]);
}
