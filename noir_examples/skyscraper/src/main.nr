use skyscraper_noir::compress;

fn main(x: u32, z: pub u32) {
  let x = x as Field;
  let z = z as Field;

  let mut r = x;
  for i in 0..1 {
    r = compress(r, r);
  }
}
