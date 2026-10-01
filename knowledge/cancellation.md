# Cancellation

## cancel-1

A renter may cancel a confirmed reservation from My trips.

## cancel-2

Cancellation at least 24 hours before `startAt` releases the vehicle and removes the rental charge. The prototype does not collect payment, so the release is recorded as status `cancelled`.

## cancel-3

Cancellation inside 24 hours of `startAt` still releases the vehicle and is marked `cancelled`. The late-cancellation note shown to the user is: one day of rental would be charged on a paid booking.
