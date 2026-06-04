# Acme Robotics — Information Security Policy (Policy ID: SEC-2026-04)

## Passwords and Authentication

All employee accounts must use a password of at least 14 characters that includes a mix
of upper-case letters, lower-case letters, numbers, and symbols. Passwords must be rotated
every 90 days. Reusing any of the previous 10 passwords is prohibited.

Multi-factor authentication (MFA) is mandatory for all systems that access customer data.
Acme Robotics uses hardware security keys (FIDO2) as the primary second factor; SMS-based
codes are not permitted for production systems. Authenticator apps are allowed only for
low-sensitivity internal tools.

Shared accounts are forbidden. Every action against a production system must be
attributable to an individual. Service accounts used by automation must have their
credentials stored in the company secrets manager and rotated automatically every 30 days.

## Data Classification

Data at Acme Robotics is classified into four tiers: Public, Internal, Confidential, and
Restricted. Restricted data includes customer personally identifiable information (PII)
and source code for the Atlas control system. Restricted data may only be stored on
company-managed devices with full-disk encryption enabled.

Confidential data includes internal financials and unreleased product plans. It may be
shared internally on a need-to-know basis but must never be sent to personal email
accounts or uploaded to unapproved cloud services. Internal data is the default tier for
day-to-day work product. Public data has been explicitly approved for external release.

Restricted data must be retained no longer than required for the stated business purpose
and securely deleted afterward. Any export of Restricted data outside company systems
requires written approval from the Data Protection Officer.

## Incident Reporting

Suspected security incidents must be reported to the Security Operations Center (SOC)
within one hour of discovery by emailing soc@acme.example or calling the 24/7 hotline.
Employees must not attempt to investigate or remediate a suspected breach themselves.

The SOC follows a four-phase incident response process: identification, containment,
eradication, and recovery. A post-incident review is completed within five business days
of resolution and documents the root cause and corrective actions.

Phishing attempts should be reported using the "Report Phish" button in the email client.
Do not forward suspicious emails to colleagues. Acme Robotics runs simulated phishing
campaigns quarterly; employees who fail two simulations in a year are assigned remedial
training.

## Device Security

Company laptops are encrypted with BitLocker (Windows) or FileVault (macOS) and lock
automatically after 5 minutes of inactivity. Lost or stolen devices must be reported to
the SOC immediately so they can be remotely wiped.

Personal devices may access company email and calendar only after enrolling in the mobile
device management (MDM) system, which enforces a screen lock and the ability to remotely
wipe company data. Jailbroken or rooted devices are blocked from all company systems.

Software may only be installed from the approved software catalog. Requests for software
outside the catalog go through the IT service desk and are reviewed for security and
licensing before approval.

## Access Reviews

Access to Restricted systems is reviewed quarterly. Managers must confirm that each team
member still requires the access they hold; unconfirmed access is automatically revoked.
When an employee leaves the company, all access is disabled within one hour of their
departure being recorded in Workday.
