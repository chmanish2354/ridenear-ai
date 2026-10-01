# Late return

## late-1

The return time is `endAt`. A return after `endAt` is late.

## late-2

The late fee is one extra rental day for any part of a day beyond `endAt`, at the same `pricePerDay` locked on the reservation.

## late-3

Repeated late returns can suspend new bookings. This prototype records the policy and does not implement suspension.
