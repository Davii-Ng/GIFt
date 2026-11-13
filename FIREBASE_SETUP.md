# Firebase account setup

## Your contribution configuration

Publish the project-root `firestore.rules` file to allow account owners to read and write metadata at `users/{uid}/contributions/{gifId}`. The updated rules preserve favorites privacy and restrict each contribution collection to its owning UID; they do not open the entire database.

In Firebase Console, select the `gift-main` project, open **Firestore Database → Rules**, replace the rules with the contents of `firestore.rules`, and click **Publish**. Reload the website and open **Your contribution** to synchronize metadata. Synchronization errors are logged only in **F12 → Console**, prefixed `[gifly:cloud-sync]`; the interface does not display banners or toasts. After fixing the configuration, reload the page to retry synchronization.

GIF files and jobs are stored persistently on the server by Firebase UID. Metadata is mirrored to the account's collection, and only public GIFs are included in shared search. See [the contribution guide](web/CONTRIBUTIONS.md) for storage settings and limits.

## Account configuration

The application uses Firebase Authentication for username/password accounts and Cloud Firestore for favorite GIFs. Guests can continue searching while Firebase configuration is pending.

1. Create a Firebase project and register a Web app in **Project settings**.
2. Copy `apiKey`, `authDomain`, `projectId`, and `appId` into `web/firebase-config.js`. These are public web configuration values. Keep service-account keys out of the frontend.
3. Keep `AUTH_USERNAME_DOMAIN` stable. Its default, `users.gifly.invalid`, is used only for internal account identifiers. Usernames are trimmed and converted to lowercase. Do not send email to these internal addresses.
4. In **Authentication → Sign-in method**, enable **Email/Password**. Under **Settings → Password policy**, select **Require** and a minimum length of 8 characters. Add deployment hostnames to **Authorized domains**, including `localhost` for local development.
5. Create Cloud Firestore in production mode.
6. From the project root, deploy the rules with the Firebase CLI:

   ```powershell
   firebase deploy --only firestore:rules --project YOUR_FIREBASE_PROJECT_ID
   ```

   Alternatively, paste `firestore.rules` into the Console's **Rules** tab and publish them there.

Restart `start-web.ps1` to serve the Firebase modules, then reload the page. Firebase Admin SDK is not required.

Successful registration signs the user in immediately. If updating `displayName` fails, the account still exists. The application logs the notification to Console and retries the display-name update on a later sign-in.

Legacy SQLite accounts and favorites are not automatically migrated to Firebase. Existing files remain under `web/.local` so the project owner can plan a migration.

## Validation with a real Firebase configuration

- Register `user_test` and confirm that the form closes and the username appears.
- Sign in as `USER_TEST` and `user_test` to confirm that both resolve to the same account.
- Save a GIF, reload the page, sign out and back in, and check from another device.
- Switch from account A to B and confirm that A's list is cleared from the interface.
- In Rules Playground or the Emulator, use UID A to read and write `users/B/favorites/test`. Both operations must be denied. Repeat as a guest.
- Attempt to write fields other than `gifId`, or a `gifId` that differs from the document ID. The rules must reject these writes.
- Disconnect before and during a save. The application must not claim success, and the control must allow another attempt.
- Disconnect during registration/sign-in and check that the form controls become available again.
- Block the SDK and confirm that guest search still works.

The following checks were completed against a real Firebase project on **2026-11-12**:

- Created two temporary accounts, updated `displayName`, and signed in successfully.
- Saved and read favorites as their owner and retrieved them after a new sign-in.
- Confirmed that rules blocked guests and account B from reading, modifying, or deleting A's favorites, and rejected mismatched `gifId` values.
- Checked automatic sign-in after registration, case-insensitive usernames, saving from a guest session, session persistence after reload, Saved navigation, and sign-out in the browser.
- Fixed two navigation issues involving sign-out from Saved and sign-in to open Saved.
- Removed the accounts and data created specifically for those checks.

Synchronization between two physical devices and actual network outages have not been verified.

## Official documentation

- [Password-based authentication](https://firebase.google.com/docs/auth/web/password-auth)
- [Authentication state persistence](https://firebase.google.com/docs/auth/web/auth-state-persistence)
- [Firestore security rule conditions](https://firebase.google.com/docs/firestore/security/rules-conditions)
