#	makefile not tested

#	alternative run with removal, STDIN open even if not attached and pseudo TTY
#       docker run --rm -it -p 8020:8010 --name app_rotate_405 rotate_form_405


build:
	docker build --tag rotate_form_405 .

run:
	docker run -p 8020:8010 --name app_rotate_405 rotate_form_405
	
update: 
	docker rm app_rotate_405
	docker run -p 8020:8010 --name app_rotate_405 rotate_form_405