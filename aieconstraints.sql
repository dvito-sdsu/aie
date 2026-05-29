
USE aie;

ALTER TABLE TreatmentLogs
ADD CONSTRAINT FK_TreatmentLogs_Users foreign key (user_id) references users(user_id);

ALTER TABLE TreatmentLogs
ADD CONSTRAINT FK_TreatmentLogs_Images foreign key (image_id) references Images(image_id);

ALTER TABLE TreatmentLogs
ADD CONSTRAINT FK_TreatmentLogs_Diseases foreign key (disease_id) references Diseases(disease_id);

ALTER TABLE TreatmentLogs
ADD CONSTRAINT FK_TreatmentLogs_Treatment foreign key (treatment_id) references Treatment(treatment_id);

ALTER TABLE images
add constraint FK_Images_Users foreign key (user_id) references Users(user_id) on delete cascade;

alter table detectionresults
add constraint FK_DetectionResults_Images foreign key (image_id) references Images(image_id);

alter table detectionresults
add constraint FK_DetectionResults_Diseases foreign key (disease_id) references Diseases(disease_id);

alter table detectionresults
add constraint FK_DetectionResults_ModelMetaData foreign key (model_id) references ModelMetaData(model_id);

alter table Treatment
add constraint FK_Treatment_Diseases foreign key (disease_id) references Diseases(disease_id);

alter table Outbreak
add constraint FK_Outbreak_Diseases foreign key (disease_id) references Diseases(disease_id);

DELIMITER $$

CREATE TRIGGER before_image_location
BEFORE INSERT ON Images
FOR EACH ROW
BEGIN -- if no image location data is provided, default to users location
    IF NEW.latitude IS NULL AND NEW.longitude IS NULL THEN
        SET NEW.latitude = (SELECT latitude FROM users WHERE user_id = NEW.user_id);
        SET NEW.longitude = (SELECT longitude FROM users WHERE user_id = NEW.user_id);
    END IF;
END$$


CREATE PROCEDURE GetUserImages(IN uid INT) -- secure method to get a users uploaded images
BEGIN
    SELECT *
    FROM Images
    WHERE user_id = uid;
END$$

CREATE PROCEDURE GetUserDetectionResults(IN uid INT) -- secure method to get a users detection results
BEGIN
    SELECT *
    FROM DetectionResults
    WHERE image_id in (select image_id from images where user_id = uid);
END$$

CREATE PROCEDURE GetActiveOutbreaks() -- quick procedure to get all active outbreaks, accessible by all
BEGIN
	select * 
	from outbreak
	WHERE last_update_date between start_date and end_date;
END$$


DELIMITER ;

-- #TODO add trigger to increment amount in outbreak table on addition to detection results