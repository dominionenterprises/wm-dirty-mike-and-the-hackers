CREATE TABLE Reviews(
	funny INT,
	user_id TEXT,
	review_id TEXT,
	text TEXT,
	business_id TEXT,
	stars INT,
	date TIMESTAMP,
	useful INT,
	type TEXT,
	cool INT
);
\copy Reviews FROM '/Users/josiah.baker/tmp.dat' CSV;