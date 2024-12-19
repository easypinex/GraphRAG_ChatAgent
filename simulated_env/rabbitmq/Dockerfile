FROM rabbitmq:3.12.8-management

# Define environment variables.
ENV RABBITMQ_PID_FILE /var/lib/rabbitmq/mnesia/rabbitmq
ENV RABBITMQ_USER user
ENV RABBITMQ_PASSWORD user123

COPY --chown=rabbitmq:rabbitmq init.sh /init.sh
# Make sure your scripts and the plugin are executable
RUN chmod +x /init.sh
USER rabbitmq
# Define default command
CMD ["/init.sh"]
