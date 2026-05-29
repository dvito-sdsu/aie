
CREATE DATABASE IF NOT EXISTS aie;
USE aie;

create table Users(
    user_id int PRIMARY KEY AUTO_INCREMENT,
    username varchar(100),
    email varchar(100) UNIQUE,
    password_hash TEXT,
    role ENUM('farmer', 'researcher', 'admin'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    longitude decimal(9,6),
    latitude decimal(8,6)
);

create table TreatmentLogs(
    log_id int PRIMARY KEY AUTO_INCREMENT,
    user_id int NOT NULL, -- fk users
    image_id int NOT NULL, -- fk images
    disease_id int, -- maybe default to -1 if unknown?
    treatment_id int NOT NULL,
    outcome ENUM('Resolved', 'No Improvements'), -- maybe add ongoing and discontinued
    notes TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

create table ModelMetadata(
    model_id int PRIMARY KEY AUTO_INCREMENT,
    version varchar(50) NOT NULL,
    framework varchar(50) NOT NULL,
    trained_date DATE,
    accuracy numeric(4,2)
);

create table Images(
    image_id int PRIMARY KEY AUTO_INCREMENT,
    user_id int,
    file_path TEXT, -- azure blob storage
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status ENUM('pending', 'processed', 'error'),
    longitude decimal(9,6) DEFAULT NULL,
    latitude decimal(8,6) DEFAULT NULL
);

create table DetectionResults(
    result_id int PRIMARY KEY AUTO_INCREMENT,
    image_id int,
    disease_id int,
    model_id int,
    confidence numeric(5,4) NOT NULL, -- might need to remove NN req
    bounding_box json, -- JSONB, JSON because error
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

create table Outbreak(
    outbreak_id INT PRIMARY KEY AUTO_INCREMENT,
    disease_id INT NOT NULL, -- fk diseases diseaseid
    longitude DECIMAL(9,6) NOT NULL,
    latitude DECIMAL(8,6) NOT NULL, 
    radius DECIMAL(5,5) NOT NULL, -- unknown, need to look into to find a good data type
    amount INT, -- will be dynamically changed based on detections and server
    start_date DATE DEFAULT (CURRENT_DATE), -- start of outbreak detection results
    end_date DATE DEFAULT ((CURRENT_DATE + INTERVAL 1 YEAR)), -- if it is determined to end or -1 for ongoing/indefinite
    last_update_date DATE DEFAULT (current_date)
);

create table Diseases(
    disease_id int PRIMARY KEY AUTO_INCREMENT,
    name varchar(50), -- 50 limit is arbitrary
    symptoms text,
    crop_type varchar(50), -- 50 limit is arbitrary
    severity enum('low','medium','high'), -- can change to float later?
    transmissibility decimal(5,5) -- arbitrary (current) value, used for radius of influence calculation
);

create table Treatment(
	treatment_id INT primary KEY auto_increment,
    disease_id int,
    description text,
    frequency varchar(50),
    source text,
    create_at datetime default current_timestamp
);
