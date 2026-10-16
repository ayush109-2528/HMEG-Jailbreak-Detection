# Connect MongoDB Atlas (free tier)

The new HMEG application uses its own `hmeg_research` database. It does not use or migrate your old Supabase project. No database is required to start the API or read the methodology pages. Dataset and experiment panels will show an unavailable state until connected; live prediction additionally requires trained artifacts.

1. Sign in at https://cloud.mongodb.com and create a project named **HMEG Research**.
2. Create a **Free** cluster. Skip sample data to conserve storage. Select an available region near your backend.
3. Under **Database Access**, create a password-authenticated database user, for example `hmeg_researcher`. Grant `readWrite` on **hmeg_research** only. Your Atlas login and database user are different accounts. Save the generated password in your password manager.
4. Under **Network Access**, add your current public IP address. When deploying, add the backend host's outbound IP addresses too. Your frontend browser does not connect to MongoDB. Avoid opening access to every IP when a narrower list is possible.
5. On the cluster select **Connect → Drivers → Python**. Copy the `mongodb+srv://...` URI.
6. In the new project's `backend` directory, copy `.env.example` to `.env` if `.env` does not already exist. Set:

   ```dotenv
   MONGODB_URI=mongodb+srv://hmeg_researcher:YOUR_ENCODED_PASSWORD@YOUR_CLUSTER.mongodb.net/?retryWrites=true&w=majority
   MONGODB_DATABASE=hmeg_research
   ```

   Replace the cluster address and password locally. Percent-encode reserved characters in the password (for example `@` becomes `%40`). Do not paste credentials into chat, source control, screenshots, or frontend environment variables. Keep the remaining settings from `.env.example`.
7. From `backend`, after installing dependencies, run:

   ```powershell
   python -m scripts.check_database
   python -m scripts.check_database --initialize
   ```

   The first command only pings the server. The second creates the application's indexes and collections; it does not delete existing documents. The script refuses the system database names `admin`, `local`, and `config`.
8. Restart the backend. Its health endpoint reports the connection state without returning credentials.

For production, configure `MONGODB_URI` and `MONGODB_DATABASE` in the **backend host** environment settings. Do not set them in Vercel's frontend. Use a separate runtime database role when you want stricter permissions; the importer needs write access to dataset collections, whereas public visitors only receive the API's bounded read access and optional consented result saving.

## Troubleshooting

- **Authentication failed:** verify the database user's password, URI encoding, and database role. Do not use your Atlas account password.
- **Connection timeout:** check the IP access list, DNS/network access, and whether the cluster is paused. Add your new IP if your internet connection changed.
- **Connected but no dataset:** connectivity does not import data or train models. Run the dataset importer separately.
- **Free storage:** the free cluster has a limited shared quota. Review the importer's size estimate and Atlas storage metrics before importing a large CSV. Existing databases on that cluster also consume space.

Official instructions: [create a free cluster](https://www.mongodb.com/docs/atlas/tutorial/deploy-free-tier-cluster/), [connect to Atlas](https://www.mongodb.com/docs/atlas/connect-to-database-deployment/), [configure access](https://www.mongodb.com/docs/atlas/security/quick-start/).
