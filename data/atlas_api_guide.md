# Atlas Control System — Developer API Guide (v3.2)

## Authentication

The Atlas API uses bearer tokens. Obtain a token by POSTing your client credentials to
`/v3/auth/token`. Tokens expire after 3600 seconds. Include the token in the
`Authorization: Bearer <token>` header on every request. The error code `ATLAS-401`
indicates an expired or invalid token.

Client credentials are issued per environment. A token minted for the staging environment
will be rejected by production with `ATLAS-403`. Rotate client secrets at least every 180
days; a leaked secret can be revoked immediately from the developer console.

## Rate Limits

The Atlas API enforces a rate limit of 120 requests per minute per client. Exceeding the
limit returns HTTP 429 with the error code `ATLAS-429` and a `Retry-After` header
indicating how many seconds to wait. Batch endpoints count as a single request regardless
of the number of items in the batch.

Rate limits are evaluated using a sliding window. Clients that consistently approach the
limit should request a quota increase through the developer console rather than retrying
aggressively, which can extend the cooldown period.

## Motion Commands

The `POST /v3/motion/move` endpoint moves a robot arm to a target pose. The request body
must include `joint_angles` (an array of six floats in radians) and an optional `speed`
field between 0.1 and 1.0, defaulting to 0.5. Sending joint angles outside the safe range
returns `ATLAS-422` and the command is rejected without moving the arm.

Each move command is acknowledged immediately with a `command_id`. The motion itself is
asynchronous; poll `GET /v3/motion/status/{command_id}` to track progress, which reports
one of `queued`, `running`, `completed`, or `failed`.

Emergency stop is triggered with `POST /v3/motion/estop`, which halts all motion within
50 milliseconds. After an emergency stop, the arm must be re-homed with
`POST /v3/motion/home` before further motion commands are accepted. The home routine takes
approximately 8 seconds and must not be interrupted.

## Gripper Control

The gripper is controlled with `POST /v3/gripper/set`, which accepts a `position` field
from 0.0 (fully closed) to 1.0 (fully open) and an optional `force` field from 0.0 to 1.0.
Closing the gripper on an object with insufficient force may cause it to slip; the
recommended default force for standard payloads is 0.6.

The gripper reports the measured grip force in its status response. A measured force that
remains near zero after a close command indicates the gripper missed the object.

## Telemetry

Telemetry is streamed over a WebSocket at `/v3/telemetry/stream`. The stream emits joint
positions, motor temperatures, and current draw at 100 Hz. If a client falls more than
500 messages behind, the server closes the connection with code 1011.

Motor temperature above 70 degrees Celsius triggers a thermal warning in the telemetry
stream; above 85 degrees the arm automatically reduces its maximum speed to protect the
motors. Sustained high current draw on a single joint often indicates a mechanical
obstruction and should be investigated before issuing further motion commands.

## Versioning and Deprecation

The Atlas API is versioned in the URL path. Breaking changes are introduced only in a new
major version. A deprecated version is supported for a minimum of 12 months after its
successor is released, and deprecation warnings are returned in the `Warning` response
header during that period.
