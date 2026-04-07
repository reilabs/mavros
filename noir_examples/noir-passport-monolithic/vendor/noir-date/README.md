# Noir Date

## Description

This library makes it easier to use date in Noir.

## Installation

In your Nargo.toml file, add the following dependency:

```
[dependencies]
date = { tag = "v0.5.4", git = "https://github.com/madztheo/noir-date.git" }
```

### Import the library

Add this line to the top of your Noir file:

```rust
use dep::date::Date;
```

## Usage

### Initialize a Date

```rust
// December 19, 2023
let date = Date::new(2023, 12, 19);

// Or alternatively from a string following this format yyyyMMdd
let date = Date::from_str_long_year("20231219");

// Or even from a byte representation of a ASCII string
let date = Date::from_bytes_long_year([50, 48, 50, 51, 49, 50, 49, 57]);
```

### Get the duration in days between two dates

```rust
let date1 = Date::new(2023, 10, 2);

let date2 = Date::new(2023, 12, 20);

// date2 - date1
let duration = date2.get_duration_in_days(date1, false);
assert(duration == 79);
```

### Add years to a date

```rust
let date = Date::new(2023, 10, 2);

let date = date.add_years(2);
assert(date.eq(Date::new(2025, 10, 2)));
```

### Add months to a date

```rust
let date = Date::new(2023, 10, 2);

let date = date.add_months(3);

assert(date.eq(Date::new(2024, 1, 2)));
```

### Add days to a date

```rust
let date = Date::new(2023, 10, 2);

let date = date.add_days(3);

assert(date.eq(Date::new(2023, 10, 5)));
```

### Check if a date is a leap year

```rust
let leap_year = Date::new(2024, 1, 1);

assert(leap_year.is_leap_year());

let not_leap_year = Date::new(2023, 1, 1);

assert(!not_leap_year.is_leap_year());
```

### Compare dates

```rust
let date1 = Date::new(2023, 10, 2);

let date2 = Date::new(2023, 12, 20);

assert(date1.lt(date2));
assert(date2.gt(date1));
```

### Check someone's age is above 18 years old

```rust
let birthdate = Date::new(1993, 3, 6);

let current_date = Date::new(2023, 12, 20);

assert(current_date.gte(birthdate.add_years(18)));
```

## Notes

This library is still in development. At the moment, there is a known bug related to comparisons with negative and positive numbers that has been fixed in the version 0.22.0 of Noir. So please install the latest nightly version of Noir to use this library.
If you find any other bugs, please report them in the issues section of this repository.
