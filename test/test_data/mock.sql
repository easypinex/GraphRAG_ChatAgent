-- ACP.dbo.file_tasks definition

-- Drop table

DROP TABLE ACP.dbo.file_tasks;

CREATE TABLE ACP.dbo.file_tasks (
	id int IDENTITY(1,1) NOT NULL,
	filedir nvarchar(256) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	filename nvarchar(256) COLLATE SQL_Latin1_General_CP1_CI_AS NOT NULL,
	status varchar(19) COLLATE SQL_Latin1_General_CP1_CI_AS NOT NULL,
	last_updated datetime NOT NULL,
	minio_dir nvarchar(256) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	user_operate varchar(6) COLLATE SQL_Latin1_General_CP1_CI_AS NULL
);

INSERT INTO ACP.dbo.file_tasks (filedir,filename,status,last_updated,minio_dir,user_operate) VALUES
	 (NULL,N'台灣人壽新住院醫療保險附約.pdf',N'COMPLETED','2024-12-18 16:03:52.897',N'user_uploaded_file/20241218_1139_9_1-台灣人壽新住院醫療保險附約.pdf',NULL);
